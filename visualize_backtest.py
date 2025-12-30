import ccxt
import pandas as pd
import numpy as np
import mplfinance as mpf
import datetime

# --- FUNCIONES DE INDICADORES ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    
    atr = true_range.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return atr

# --- DESCARGA DE DATOS ---
def fetch_data_bulk(symbol='BTC/USDT', timeframe='4h', limit=5000):
    print(f"--- Descargando datos para {symbol} ({timeframe}) ---")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Calcular cuánto tiempo atrás necesitamos
    # 5000 velas * 4 horas = 20,000 horas
    # 20,000 / 24 = 833.33 días
    days_history = (limit * 4) / 24 + 20 # buffer extra
    
    now = exchange.milliseconds()
    since = int(now - (days_history * 24 * 60 * 60 * 1000))
    
    all_ohlcv = []
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=int(since), limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1
            
            if len(all_ohlcv) >= limit + 500: # Descargar un poco más para indicadores
                break
                
        except Exception as e:
            print(f"Error descargando: {e}")
            break
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=False) # mplfinance necesita indice datetime
    df.index = pd.DatetimeIndex(df['timestamp'])
    
    # Recortar a las últimas N velas solicitadas pero asegurando tener previo para indicadores
    # Calculamos indicadores sobre todo y recortamos DESPUÉS para visualización
    return df

def main():
    # 1. Configuración Ganadora
    EMA_PERIOD = 200
    RSI_LIMIT = 35
    RR_RATIO = 2.0
    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '4h'
    SOLICITED_CANDLES = 5000
    
    # 2. Descargar Datos
    df = fetch_data_bulk(SYMBOL, TIMEFRAME, limit=SOLICITED_CANDLES)
    
    if df is None or df.empty:
        print("No se pudieron descargar los datos.")
        return

    print(f"Datos descargados: {len(df)} velas.")

    # 3. Calcular Indicadores
    print("Calculando indicadores...")
    df['EMA_200'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['RSI_14'] = calculate_rsi(df['close'], period=14)
    df['prev_RSI'] = df['RSI_14'].shift(1)
    df['ATR_14'] = calculate_atr(df, period=14)
    
    # Recortar a las últimas 5000 para el análisis/plot (dejando un buffer previo para que la EMA no empiece en 0)
    # Pero si cortamos muy agresivo, la EMA se vería mal al principio.
    # Mejor usaremos todo el dataframe para calculo y slicearemos para plot.
    
    # 4. Simular Lógica de Trading para generar marcadores
    print("Simulando entradas y salidas...")
    
    buy_signals = [np.nan] * len(df)
    tp_hits = [np.nan] * len(df)
    sl_hits = [np.nan] * len(df)
    
    in_position = False
    position = {}
    
    # Lógica de entrada: Trend + Trigger
    trend_filter = df['close'] > df['EMA_200']
    # Trigger: RSI cruza por debajo de 35 (Entrada en dip/pullback agresivo segun logica anterior)
    trigger = (df['RSI_14'] < RSI_LIMIT) & (df['prev_RSI'] >= RSI_LIMIT)
    signals = trend_filter & trigger
    
    # Iterar con índice numérico para rellenar listas
    # df.index es Datetime, usaremos rango numerico
    for i in range(len(df)):
        # Acceso rápido por posición
        current_close = df['close'].iloc[i]
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        current_open = df['open'].iloc[i]
        current_atr = df['ATR_14'].iloc[i]
        
        # Gestión de Posición
        if in_position:
            sl = position['sl']
            tp = position['tp']
            
            sl_hit = current_low <= sl
            tp_hit = current_high >= tp
            
            if sl_hit:
                # Prioridad a SL (ser conservador o si ambos ocurren, asumimos SL)
                sl_hits[i] = sl # Marcamos en el nivel de SL
                in_position = False
            elif tp_hit:
                tp_hits[i] = tp # Marcamos en el nivel de TP
                in_position = False
        
        # Entrada (solo si no estamos en posición)
        if not in_position:
            if signals.iloc[i]:
                # Validar que tengamos datos suficientes (EMA no nula)
                if not pd.isna(df['EMA_200'].iloc[i]):
                    entry_price = current_close
                    
                    sl_dist = 2.0 * current_atr
                    tp_dist = sl_dist * RR_RATIO
                    
                    sl = entry_price - sl_dist
                    tp = entry_price + tp_dist
                    
                    position = {'sl': sl, 'tp': tp, 'entry': entry_price}
                    buy_signals[i] = entry_price # Flecha en el precio de entrada
                    in_position = True

    # 5. Filtrar para Plotting (Últimas 500 velas para que se vea bien, o 5000 si se pide todo)
    # 5000 velas en una imagen estática es muy denso.
    # El usuario pidió "últimas 5000 velas" y "visualizar".
    # Generaremos la imagen completa pero advertiremos que es grande.
    # Para mejor visualización, haremos un slice de las últimas 500 velas donde hubo acción.
    # Pero el usuario pidió audit, así que tal vez quiera ver TODO.
    # Haremos el plot de las últimas 500 para que se vea CLARO el detalle de las velas.
    # O mejor: plotear todo guardando en un archivo de alta resolución.
    
    plot_limit = 5000
    if len(df) > plot_limit:
        df_plot = df.iloc[-plot_limit:]
        buy_plot = buy_signals[-plot_limit:]
        sl_plot = sl_hits[-plot_limit:]
        tp_plot = tp_hits[-plot_limit:]
    else:
        df_plot = df
        buy_plot = buy_signals
        sl_plot = sl_hits
        tp_plot = tp_hits

    # Configurar AddPlots
    apds = [
        mpf.make_addplot(df_plot['EMA_200'], color='blue', width=1.5),
        mpf.make_addplot(buy_plot, type='scatter', markersize=100, marker='^', color='green', label='Buy'),
        mpf.make_addplot(sl_plot, type='scatter', markersize=100, marker='o', color='red', label='SL Hit'),
        mpf.make_addplot(tp_plot, type='scatter', markersize=100, marker='o', color='lime', label='TP Hit'),
    ]

    print("Generando gráfico (esto puede tomar unos segundos)...")
    
    # Estilo visual
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 10})
    
    mpf.plot(
        df_plot,
        type='candle',
        style=s,
        addplot=apds,
        title=f"BTC/USDT Backtest Audit\nEMA: {EMA_PERIOD} | RSI: {RSI_LIMIT} | R:R: {RR_RATIO}",
        ylabel='Price (USDT)',
        volume=False, # Volumen ensucia si hay muchas velas
        figsize=(24, 12), # Imagen grande
        savefig=dict(fname='backtest_result.png', dpi=300, bbox_inches='tight')
    )
    
    print("¡Listo! Gráfico guardado en 'backtest_result.png'.")

if __name__ == "__main__":
    main()
