import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys

# WINDOWS COMPATIBILITY (UTF-8)
# Vital para evitar crash en Windows al imprimir emojis
sys.stdout.reconfigure(encoding='utf-8')

# CONFIGURACIÓN
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1h'
LIMIT = 500  # Velas necesarias para EMA 200
LOG_FILE = 'trading_log.txt'

# Inicializar Exchange (Datos públicos)
exchange = ccxt.binance({
    'enableRateLimit': True,
})

def log_signal(message):
    """Guarda mensajes importantes en el archivo de log con codificación UTF-8."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # CORRECCIÓN DE ENCODING: encoding='utf-8' para soportar emojis en Windows
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")

def calculate_manual_indicators(df):
    """Calcula EMA, RSI y ADX manualmente sin pandas_ta."""
    # EMA 200
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # RSI 14 (Wilder's Smoothing)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Wilder's Smoothing: alpha = 1/n
    alpha = 1/14
    avg_gain = gain.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # ADX 14
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # DM+ y DM-
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    # Smoothed components
    atr = tr.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, min_periods=14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=14, adjust=False).mean() / atr)
    
    # DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX (Smoothed DX)
    df['ADX'] = dx.ewm(alpha=alpha, min_periods=14, adjust=False).mean()
    
    return df

def fetch_data():
    """Descarga datos de velas y calcula indicadores."""
    try:
        # Descargar OHLCV
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=LIMIT)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Calcular Indicadores Manualmente
        df = calculate_manual_indicators(df)
        
        return df
    except Exception as e:
        print(f"Error descargando datos: {e}")
        return None

def main():
    print(f"=== INICIANDO BOT: {SYMBOL} ({TIMEFRAME}) ===")
    print("Estrategia: 'La Mina de Oro' (Trend Filter EMA200 + RSI Dip)")
    log_signal(f"=== BOT INICIADO ({SYMBOL} {TIMEFRAME}) ===")
    
    while True:
        try:
            # 1. Obtener Datos
            df = fetch_data()
            
            if df is None or df.empty:
                time.sleep(10)
                continue
                
            # Usamos la ÚLTIMA VELA CERRADA para análisis técnico
            last_closed = df.iloc[-2]
            
            # Obtener precio actual real del ticker para display
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            
            # Valores de Análisis (Vela Cerrada)
            close_price = last_closed['close']
            ema_200 = last_closed['EMA_200']
            rsi = last_closed['RSI']
            adx = last_closed['ADX']
            
            # 2. Lógica de Decisión
            
            # PASO 1: FILTRO DE TENDENCIA (El Escudo)
            trend = "BAJISTA [NO]"
            trend_ok = False
            
            if close_price > ema_200:
                trend = "ALCISTA [OK]"
                trend_ok = True
                
            # PASO 2: GATILLO (RSI + ADX)
            # RSI < 50 Y ADX > 20
            signal = "ESPERANDO..."
            trigger_triggered = False
            
            if trend_ok:
                if rsi < 50 and adx > 20:
                    signal = "!!! COMPRA !!! [BUY]"
                    trigger_triggered = True
                else:
                    signal = "Esperando Dip/Confirmacion"
            else:
                signal = "MERCADO BAJISTA - PROHIBIDO OPERAR"

            # 3. Output Consola (Diseño Limpio)
            print("-" * 40)
            print(f"Fecha:     {datetime.now().strftime('%H:%M:%S')}")
            print(f"Precio:    ${current_price:,.2f}")
            print(f"Tendencia: {trend} (Close: {close_price:.1f} vs EMA: {ema_200:.1f})")
            print(f"RSI (14):  {rsi:.2f}")
            print(f"ADX (14):  {adx:.2f}")
            print(f"ESTADO:    {signal}")
            print("-" * 40)
            
            # 4. Logs
            if trigger_triggered:
                print(">> SEÑAL DE COMPRA DETECTADA <<")
                log_signal(f"BUY SIGNAL | Price: {current_price} | RSI: {rsi:.2f} | ADX: {adx:.2f}")
            
            # Esperar 60 segundos antes de la siguiente revisión
            time.sleep(5)
            
        except KeyboardInterrupt:
            print("\nBot detenido por usuario.")
            break
        except Exception as e:
            print(f"Error inesperado en main loop: {e}")
            time.sleep(10) # Esperar antes de reintentar

if __name__ == "__main__":
    main()
