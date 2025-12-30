import ccxt
import pandas as pd
import numpy as np
import time

# --- FUNCIONES DE INDICADORES (Manuales por fallback) ---

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    # Wilder's Smoothing for RSI
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

def calculate_adx(df, period=14):
    """
    Calcula el Average Directional Index (ADX) usando Wilder's Smoothing.
    Replica la logica estandar de ADX.
    """
    # 1. Calcular TR, +DM, -DM
    df['tr0'] = np.abs(df['high'] - df['low'])
    df['tr1'] = np.abs(df['high'] - df['close'].shift(1))
    df['tr2'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0.0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0.0)
    
    # 2. Suavizar TR, +DM, -DM (Wilder's Smoothing: alpha = 1/period)
    alpha = 1 / period
    df['atr_smooth'] = df['tr'].ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    df['plus_di_smooth'] = df['plus_dm'].ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    df['minus_di_smooth'] = df['minus_dm'].ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    
    # 3. Calcular +DI y -DI
    df['plus_di'] = 100 * (df['plus_di_smooth'] / df['atr_smooth'])
    df['minus_di'] = 100 * (df['minus_di_smooth'] / df['atr_smooth'])
    
    # 4. Calcular DX
    sum_di = df['plus_di'] + df['minus_di']
    diff_di = np.abs(df['plus_di'] - df['minus_di'])
    df['dx'] = 100 * (diff_di / sum_di)
    
    # 5. Calcular ADX (Suavizado del DX)
    df['adx'] = df['dx'].ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    
    return df['adx']

# --- DESCARGA DE DATOS MASIVA ---
def fetch_data_bulk(symbol='BTC/USDT', timeframe='4h', limit=5000):
    print(f"--- Descargando datos para {symbol} ({timeframe}) ---")
    exchange = ccxt.binance({'enableRateLimit': True})
    
    days_history = (limit * 4) / 24 + 50 # buffer
    now = exchange.milliseconds()
    since = int(now - (days_history * 24 * 60 * 60 * 1000))
    
    all_ohlcv = []
    
    while since < now:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break
                
            all_ohlcv += ohlcv
            since = ohlcv[-1][0] + 1
            
            if len(all_ohlcv) >= limit + 200:
                break
                
        except Exception as e:
            print(f"Error descargando: {e}")
            break

    if not all_ohlcv:
        return None
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    cols = ['open', 'high', 'low', 'close', 'volume']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    
    df.drop_duplicates(subset=['timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Total velas descargadas: {len(df)}")
    return df

# --- PRE-CÁLCULO DE INDICADORES ---
def prepare_indicators(df):
    if df is None or df.empty: return None
    print("--- Calculando Indicadores (EMA, RSI, ADX, ATR) ---")
    
    # 1. EMA Fixed (200)
    df['EMA_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # 2. RSI Fixed (14)
    df['RSI_14'] = calculate_rsi(df['close'], period=14)
    df['prev_RSI'] = df['RSI_14'].shift(1)
    
    # 3. ADX (14)
    df['ADX_14'] = calculate_adx(df.copy(), period=14)
    
    # 4. ATR (14) para SL/TP
    df['ATR_14'] = calculate_atr(df, period=14)
    
    # Limpieza
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- BACKTEST CORE (Optimizado con ADX) ---
def run_backtest_logic(df, adx_threshold, initial_balance=10000.0):
    
    # Parámetros FIJOS ganadores
    EMA_PERIOD = 200
    RSI_LIMIT = 35
    RR_RATIO = 2.0
    
    balance = initial_balance
    in_position = False
    position = {} 
    
    wins = 0
    losses = 0
    total_trades = 0
    
    gross_profit = 0.0
    gross_loss = 0.0
    
    # --- Vectorización de Señales ---
    # 1. Trend: Precio > EMA 200
    cond_trend = df['close'] > df['EMA_200']
    
    # 2. Trigger: RSI < 35 y RSI anterior >= 35 (Cruce bajista = Entrada en dip)
    cond_trigger = (df['RSI_14'] < RSI_LIMIT) & (df['prev_RSI'] >= RSI_LIMIT) 
    
    # 3. FILTRO ADX: ADX > Umbral (Tendencia fuerte)
    cond_adx = df['ADX_14'] > adx_threshold
    
    # SEÑAL FINAL
    signals = cond_trend & cond_trigger & cond_adx
    
    # --- Iteración de Portfolio ---
    for row in df.itertuples():
        
        # Gestión de Posición
        if in_position:
            sl = position['sl']
            tp = position['tp']
            
            # Verificar hits (Conservador: SL primero en la misma vela)
            if row.low <= sl:
                exit_price = sl
                pnl = (exit_price - position['entry_price']) * position['size']
                balance += pnl
                losses += 1
                gross_loss += abs(pnl)
                in_position = False
                total_trades += 1
            
            elif row.high >= tp:
                exit_price = tp
                pnl = (exit_price - position['entry_price']) * position['size']
                balance += pnl
                wins += 1
                gross_profit += pnl
                in_position = False
                total_trades += 1
        
        # Entrada
        if not in_position:
            # Usar señales pre-calculadas
            if signals[row.Index]:
                atr = row.ATR_14
                entry_price = row.close
                
                # Gestión de Riesgo: 2% por operación
                risk_amt = balance * 0.02
                
                sl_dist = 2.0 * atr # SL Fijo relativo a ATR
                tp_dist = sl_dist * RR_RATIO # TP basado en RR ganador
                
                if sl_dist > 0:
                    size = risk_amt / sl_dist
                    position = {
                        'entry_price': entry_price,
                        'size': size,
                        'sl': entry_price - sl_dist,
                        'tp': entry_price + tp_dist
                    }
                    in_position = True
    
    # Métricas Finales
    net_profit = balance - initial_balance
    net_profit_pct = (net_profit / initial_balance) * 100
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float('inf') if gross_profit > 0 else 0
    
    return {
        'ADX Threshold': adx_threshold,
        'Net Profit %': round(net_profit_pct, 2),
        'Win Rate %': round(win_rate, 2),
        'Profit Factor': round(profit_factor, 2),
        'Total Trades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'Final Balance': round(balance, 2)
    }

def optimize_adx(df):
    print("\n--- Iniciando Optimización de Filtro ADX ---")
    print("Parámetros Fijos: EMA=200, RSI<35, RR=2.0")
    
    adx_params = [20, 25, 30, 35]
    results = []
    
    print(f"Probando umbrales ADX: {adx_params}...")
    
    for adx_val in adx_params:
        res = run_backtest_logic(df, adx_val)
        results.append(res)
        
    results_df = pd.DataFrame(results)
    
    # Ordenar por Profit Factor
    results_df = results_df.sort_values(by='Profit Factor', ascending=False)
    
    return results_df

def main():
    SYMBOL = 'BTC/USDT'
    TIMEFRAME = '4h'
    LIMIT = 5000 
    
    # 1. Obtener Datos
    df = fetch_data_bulk(SYMBOL, TIMEFRAME, LIMIT)
    
    if df is not None:
        # 2. Preparar Indicadores
        df = prepare_indicators(df)
        
        # 3. Ejecutar Comparativa ADX
        results = optimize_adx(df)
        
        print("\n--- RESULTADOS DE LA OPTIMIZACIÓN ADX ---")
        cols = ['ADX Threshold', 'Net Profit %', 'Win Rate %', 'Profit Factor', 'Total Trades', 'Wins', 'Losses']
        print(results[cols].to_string(index=False))
        
        # Recomendación
        if not results.empty:
            best = results.iloc[0]
            print(f"\nCONCLUSIÓN: El mejor umbral es ADX > {best['ADX Threshold']}")
            print(f"Profit Factor: {best['Profit Factor']} | Win Rate: {best['Win Rate %']}%")

if __name__ == "__main__":
    main()
