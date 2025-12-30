import ccxt
import pandas as pd
import numpy as np
import time
import os
import csv
import sys
import io
from datetime import datetime
import requests # Nuevo import

# Force UTF-8 stdout for Windows
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Nuevos imports de la librería 'ta'
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

# --- CONFIGURACIÓN ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
EMA_PERIOD = 200
RSI_PERIOD = 14
RSI_BUY_LIMIT = 35
ADX_PERIOD = 14
ADX_THRESHOLD = 20
ATR_PERIOD = 14
TAKE_PROFIT_PCT = 0.012
STOP_LOSS_PCT = 0.008
LOG_FILE = 'bitacora_trading.csv'

# --- CONFIGURACIÓN TELEGRAM ---

TELEGRAM_TOKEN = "PON_AQUI_TU_TOKEN"
TELEGRAM_ID = "PON_AQUI_TU_ID"

def enviar_telegram(mensaje):
    """Envía un mensaje a Telegram. No detiene el bot si falla."""
    if TELEGRAM_TOKEN == "PON_AQUI_TU_TOKEN" or TELEGRAM_ID == "PON_AQUI_TU_ID":
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": TELEGRAM_ID, "text": mensaje}
    
    try:
        requests.post(url, data=data, timeout=5)
    except Exception as e:
        print(f"⚠️ Error enviando Telegram: {e}")

# --- GESTOR DE ESTADO Y LOGGING ---
class PaperTradingBot:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        
        # Billetera Virtual
        self.saldo_usdt = 1000.0   # Capital inicial
        self.btc_tenencia = 0.0    # BTC en posesión
        self.en_posicion = False   # Estado
        self.precio_compra = 0.0   # Precio de entrada real
        
        self.setup_logging()
        
    def setup_logging(self):
        file_exists = os.path.exists(LOG_FILE)
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Fecha', 'Tipo', 'Precio', 'Detalles', 'Resultado', 'Saldo USDT'])

    def log_event(self, event_type, price, details, result=''):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            current_balance = self.saldo_usdt if not self.en_posicion else (self.btc_tenencia * price)
            writer.writerow([timestamp, event_type, price, details, result, f"{current_balance:.2f}"])
        print(f"[{timestamp}] {event_type}: {details}")

    def fetch_ohlcv(self):
        # Descargamos suficientes velas para la EMA 200
        # Usamos try-except para robustez de red
        try:
            ohlcv = self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=300)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"⚠️ Error descarga OHLCV: {e}")
            return pd.DataFrame()

    def update_indicators(self, df):
        if df.empty: return df
        
        # 1. EMA 200
        ema_indicator = EMAIndicator(close=df['close'], window=EMA_PERIOD)
        df['EMA_200'] = ema_indicator.ema_indicator()
        
        # 2. RSI 14
        rsi_indicator = RSIIndicator(close=df['close'], window=RSI_PERIOD)
        df['RSI_14'] = rsi_indicator.rsi()
        
        # 3. ADX 14
        adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=ADX_PERIOD)
        df['ADX_14'] = adx_indicator.adx()
        
        return df

    def get_current_price(self):
        try:
            ticker = self.exchange.fetch_ticker(SYMBOL)
            return float(ticker['last'])
        except Exception as e:
            print(f"⚠️ Error obteniendo precio: {e}")
            return None

    def execute_buy(self):
        # IMPORTANTE: Obtener precio fresco para ejecución realista
        execution_price = self.get_current_price()
        if execution_price is None:
            print("❌ No se pudo obtener precio para comprar.")
            return

        if self.saldo_usdt <= 0:
            print("❌ Saldo insuficiente para comprar.")
            return

        print(f"\n🔵 EJECUTANDO COMPRA (Market) a ${execution_price:.2f}...")
        
        # Calcular monto
        gross_usdt = self.saldo_usdt
        fee = gross_usdt * 0.001  # 0.1% comisión
        net_usdt = gross_usdt - fee
        
        cantidad_btc = net_usdt / execution_price
        
        # Actualizar Billetera
        self.btc_tenencia = cantidad_btc
        self.saldo_usdt = 0.0
        self.precio_compra = execution_price
        self.en_posicion = True
        
        details = f"Comprado {cantidad_btc:.6f} BTC | Fee: ${fee:.2f}"
        self.log_event("COMPRA", execution_price, details)
        
        msg_telegram = f"🚀 COMPRA SIMULADA ({SYMBOL})\n\nPrecio: ${execution_price:.2f}\nCant: {cantidad_btc:.6f} BTC\nSaldo Restante: ${self.saldo_usdt:.2f}"
        enviar_telegram(msg_telegram)
        
        print(f"✅ COMPRA COMPLETADA | Tenencia: {self.btc_tenencia:.6f} BTC | Precio Entrada: ${self.precio_compra:.2f}")

    def execute_sell(self, cause_msg):
        # IMPORTANTE: Obtener precio fresco para ejecución realista
        execution_price = self.get_current_price()
        if execution_price is None:
            print("❌ No se pudo obtener precio para vender.")
            return

        if self.btc_tenencia <= 0:
            return

        print(f"\n{cause_msg} - EJECUTANDO VENTA (Market) a ${execution_price:.2f}...")
        
        gross_usdt = self.btc_tenencia * execution_price
        fee = gross_usdt * 0.001 # 0.1% comisión
        net_usdt = gross_usdt - fee
        
        # Calculo de Ganancia Real del Trade
        # Ganancia = Saldo Final - Saldo Inicial (antes de comprar)
        # Reconstruimos saldo inicial con el precio de compra y la cantidad
        # Aproximación: prev_wealth ~ (btc_tenencia * precio_compra) / 0.999 
        # Pero simplifiquemos: PnL = net_usdt - (precio_compra * btc_tenencia) 
        # (Esto ignora el fee de entrada en el cálculo de 'Ganancia del Trade' mostrado, 
        # pero el 'Nuevo Saldo' SI es correcto y es lo importante)
        
        costo_bruto_compra = self.precio_compra * self.btc_tenencia 
        ganancia_trade_usd = net_usdt - costo_bruto_compra
        
        self.saldo_usdt = net_usdt
        self.btc_tenencia = 0.0
        self.en_posicion = False
        self.precio_compra = 0.0
        
        details = f"Vendido a {execution_price:.2f} | Fee: ${fee:.2f}"
        res_str = f"Nuevo Saldo: ${self.saldo_usdt:.2f}"
        
        self.log_event("VENTA", execution_price, details, res_str)
        
        icon = "✅" if ganancia_trade_usd >= 0 else "❌"
        msg_telegram = f"{icon} VENTA SIMULADA ({SYMBOL})\n\nTipo: {cause_msg}\nPrecio Salida: ${execution_price:.2f}\nResultado Trade: ${ganancia_trade_usd:.2f}\nNuevo Saldo: ${self.saldo_usdt:.2f}"
        enviar_telegram(msg_telegram)

        print(f"💰 VENTA EJECUTADA | Ganancia: ${ganancia_trade_usd:.2f} | Nuevo Saldo: ${self.saldo_usdt:.2f}")


    def run(self):
        print(f"🚀 Bot iniciado en 1H - Modo Intraday ({SYMBOL})")
        print(f"💰 Capital Inicial: ${self.saldo_usdt:.2f} USDT")
        print(f"📊 Estrategia: EMA {EMA_PERIOD}, RSI < {RSI_BUY_LIMIT}, ADX > {ADX_THRESHOLD}")
        print(f"🎯 Salidas: TP +{TAKE_PROFIT_PCT*100:.1f}% | SL -{STOP_LOSS_PCT*100:.1f}% (sobre precio real de compra)")
        
        last_processed_candle_time = None
        
        while True:
            try:
                # 1. Obtener Precio Real-Time para monitoreo
                current_price = self.get_current_price()
                if current_price is None:
                    time.sleep(5)
                    continue

                # 2. Monitoreo de Salida (SL/TP) - Priorities
                if self.en_posicion:
                    change_pct = (current_price - self.precio_compra) / self.precio_compra
                    pnl_display = change_pct * 100
                    
                    # Feedback Visual en Tiempo Real (Live Ticker) - CON POSICIÓN
                    sys.stdout.write(f"\r🟢 OPERACIÓN ACTIVA | Entrada: ${self.precio_compra:.2f} | Actual: ${current_price:.2f} | PnL: {pnl_display:+.2f}%   ")
                    sys.stdout.flush()

                    if change_pct >= TAKE_PROFIT_PCT:
                        sys.stdout.write("\n") # Limpiar linea
                        self.execute_sell("🟢 TAKE PROFIT")
                        continue 
                    
                    elif change_pct <= -STOP_LOSS_PCT:
                        sys.stdout.write("\n") # Limpiar linea
                        self.execute_sell("🔴 STOP LOSS")
                        continue 
                else:
                    # Feedback Visual en Tiempo Real (Live Ticker) - SIN POSICIÓN
                    sys.stdout.write(f"\r⏳ Esperando cierre vela... Precio: ${current_price:.2f}   ")
                    sys.stdout.flush()
                
                # 3. Monitoreo de Entrada (Solo si no estamos comprados)
                if not self.en_posicion:
                    # Optimización: Consultamos OHLCV, pero la lógica impide recalcular si es la misma vela
                    df = self.fetch_ohlcv()
                    if not df.empty:
                        last_candle = df.iloc[-2] # Vela cerrada
                        last_candle_time = last_candle['timestamp']
                        
                        # Solo analizamos si cerramos una nueva vela
                        if last_processed_candle_time != last_candle_time:
                            sys.stdout.write("\n") # Salto de linea para que el log no pise el ticker
                            print(f"🔎 Nueva vela cerrada: {last_candle_time} | Close: {last_candle.close}")
                            last_processed_candle_time = last_candle_time
                            
                            df = self.update_indicators(df)
                            last_candle = df.iloc[-2]
                            
                            if pd.notna(last_candle.EMA_200) and pd.notna(last_candle.RSI_14) and pd.notna(last_candle.ADX_14):
                                
                                is_trend = last_candle.close > last_candle.EMA_200
                                is_dip = last_candle.RSI_14 < RSI_BUY_LIMIT
                                is_strong = last_candle.ADX_14 > ADX_THRESHOLD
                                
                                print(f"   Métricas: RSI={last_candle.RSI_14:.1f} | ADX={last_candle.ADX_14:.1f} | EMA={last_candle.EMA_200:.1f}")
                                
                                if is_trend and is_dip and is_strong:
                                    print("   ✅ SEÑAL DE ENTRADA VALIDADA")
                                    self.execute_buy()
                                else:
                                    print("   ❌ Sin señal.")
                            else:
                                print("   ⚠️ Cargando datos históricos...")

                # Heartbeat más rápido (5 segundos)
                time.sleep(5) 
                
            except Exception as e:
                # Usamos \n para separar el error del ticker
                sys.stdout.write("\n")
                print(f"⚠️ Error en el bucle principal: {e}")
                time.sleep(5)

if __name__ == "__main__":
    bot = PaperTradingBot()
    bot.run()
