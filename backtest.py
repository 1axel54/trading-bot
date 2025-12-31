import yfinance as yf
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    pass

def run_backtest():
    # 1. Configuración de datos
    ticker = 'BTC-USD'
    interval = '1h'
    period = '3y'
    initial_capital = 100000.0
    
    # Parámetros de Estrategia y Riesgo
    RISK_PER_TRADE = 0.005  # 0.5% riesgo por operación
    RSI_BUY_THRESHOLD = 50  # RSI < 30 para entrar
    SL_PCT = 0.01           # 1% Stop Loss
    TP_PCT = 0.02           # 2% Take Profit
    
    # 2. Descargar datos
    print(f"Descargando datos para {ticker}...")
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if df.empty:
        print("Error: No se descargaron datos.")
        return

    # Aplanar MultiIndex si es necesario (yfinance devuelve MultiIndex a veces)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 3. Calcular Indicadores
    try:
        # Intentar usar pandas_ta si está instalado
        df['RSI'] = df.ta.rsi(length=14)
        df['EMA_200'] = df.ta.ema(length=200)
        adx_df = df.ta.adx(length=14)
        if 'ADX_14' in adx_df.columns:
            df['ADX'] = adx_df['ADX_14']
        else:
            df = pd.concat([df, adx_df], axis=1)
            df.rename(columns={'ADX_14': 'ADX'}, inplace=True)
    except (ImportError, AttributeError):
        print("pandas_ta no encontrado o error. Usando cálculo manual de indicadores...")
        
        # RSI Manual
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean() # SMMA es mejor pero rolling simple para fallback
        # Para ser más precisos con Wilder (pandas_ta default):
        # alpha = 1/14
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ADX Manual
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
        
        # Smoothed
        atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        
        # EMA 200 Manual
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Limpiar NaN iniciales generados por indicadores
    df.dropna(inplace=True)

    # 4. Simulación Loop
    balance = initial_capital
    peak_equity = initial_capital
    max_drawdown = 0.0
    position = None # None, o diccionario {'entry_price': float, 'amount': float}
    trades = [] # Lista de resultados de operaciones {'result': 'win'/'loss', 'pnl': float}
    
    print("Iniciando simulación...")
    
    # Iteramos por el índice (timestamps) y usamos iloc para acceder a posiciones
    # Convertimos a np array o iterrows es lento, pero para 60d * 24h = 1440 velas, iterrows es aceptable.
    
    # Necesitamos acceso secuencial.
    # Usaremos iterrows para recorrer vela a vela.
    
    for index, row in df.iterrows():
        current_price = row['Close']
        high = row['High']
        low = row['Low']
        rsi = row['RSI']
        rsi = row['RSI']
        adx = row['ADX']
        ema_200 = row['EMA_200']
        
        # Calcular Equity Actual y Drawdown
        current_equity = balance
        if position:
            current_equity = balance + (current_price - position['entry_price']) * position['amount']
            
        if current_equity > peak_equity:
            peak_equity = current_equity
            
        drawdown = (peak_equity - current_equity) / peak_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        
        # Lógica de Salida (SI estamos comprados)
        if position is not None:
            entry_price = position['entry_price']
            amount = position['amount']
            
            # Definir niveles
            tp_price = entry_price * (1 + TP_PCT)
            sl_price = entry_price * (1 - SL_PCT)
            
            # Verificar si en esta vela el precio tocó SL o TP
            # Asumimos que verificamos contra High/Low de la vela ACTUAL
            # NOTA: En backtesting riguroso, entraríamos al CIERRE de la vela anterior, 
            # así que chequeamos TP/SL en la vela ACTUAL (High/Low).
            
            # Prioridad: Ser conservador. Si Low low toca SL, asumimos SL primero.
            if low <= sl_price:
                # Stop Loss ejecutado
                exit_price = sl_price
                pnl = (exit_price - entry_price) * amount
                balance += pnl
                trades.append({'result': 'LOSS', 'pnl': pnl, 'exit_price': exit_price})
                position = None
                continue # Posición cerrada, pasamos a siguiente vela (no compramos en la misma que cerramos por convención simple)

            if high >= tp_price:
                # Take Profit ejecutado
                exit_price = tp_price
                pnl = (exit_price - entry_price) * amount
                balance += pnl
                trades.append({'result': 'WIN', 'pnl': pnl, 'exit_price': exit_price})
                position = None
                continue

        # Lógica de Entrada (SI NO estamos comprados)
        if position is None:
            # Condición: RSI < 30 Y ADX > 25 Y Price > EMA 200
            if rsi < RSI_BUY_THRESHOLD and adx > 25 and current_price > ema_200:
                # COMPRA
                # Gestión de Riesgo Dinámica
                # Risk Amount = Balance * RISK_PER_TRADE
                # Position Size = Risk Amount / SL_PCT
                risk_amount = balance * RISK_PER_TRADE
                amount = risk_amount / (current_price * SL_PCT)
                
                # Verificación de fondos (no apalancamiento aqui)
                cost = amount * current_price
                if cost > balance:
                    amount = balance / current_price

                position = {
                    'entry_price': current_price,
                    'amount': amount
                }
                # No descontamos spread/comisión según instrucciones, pero se podría añadir.
                
    # 5. Resultados
    total_trades = len(trades)
    wins = len([t for t in trades if t['result'] == 'WIN'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    net_pnl = balance - initial_capital
    
    print("\n" + "="*30)
    print("RESUMEN FINAL DE BACKTEST")
    print("="*30)
    print(f"Capital Inicial:   ${initial_capital:,.2f}")
    print(f"Capital Final:     ${balance:,.2f}")
    print(f"Total Operaciones: {total_trades}")
    print(f"Win Rate:          {win_rate:.2f}%")
    print(f"Ganancia/Pérdida:  ${net_pnl:,.2f}")
    print(f"Max Drawdown:      {max_drawdown*100:.2f}%")
    if max_drawdown > 0.10:
        print("ALERTA: El MDD ha superado el 10% permitidos!")
    print("="*30)

if __name__ == "__main__":
    run_backtest()
