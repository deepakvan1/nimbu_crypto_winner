from binance.error import ClientError
import pandas as pd
import pandas_ta as ta
import numpy as np
from .models import Trade
from django.conf import settings

# Strategy parameters
RISK_PERCENT = 0.01  # 1% risk per trade
REWARD_RATIO = 3.0  # Risk:Reward ratio
EMA_FAST = 9
EMA_SLOW = 21
VOLUME_PERIOD = 20
VOLUME_THRESHOLD = 1.5
MOMENTUM_THRESHOLD = 0.1  # 0.1%
TREND_STRENGTH_THRESHOLD = 0.05  # 0.05%

def fetch_historical_data(client_obj, symbol, interval, limit=1000):
    try:
        resp = pd.DataFrame(client_obj.klines(symbol, interval, limit=limit))
        resp = resp.iloc[:, :6]  # Keep only OHLCV columns
        resp.columns = ['Time', 'open', 'high', 'low', 'close', 'volume']
        resp = resp.set_index('Time')
        resp.index = pd.to_datetime(resp.index, unit='ms')
        resp = resp.astype(float)
        return resp
    except ClientError as error:
        print(f"Error fetching data for {symbol}: {error.error_message}")
        return None

def generate_trading_signals(df):
    """
    Generate trading signals based on the Pine Script strategy (EMA, Bollinger Bands, Supertrend).
    
    Args:
        df: 15-minute OHLCV data
    
    Returns:
        DataFrame with trading signals, entry/exit levels, and side information
    """
    df = df.copy()
    df['time'] = df.index
    df.reset_index(drop=True, inplace=True)
    
    # Calculate technical indicators
    df['ema_fast'] = ta.ema(df['close'], length=EMA_FAST)
    df['ema_slow'] = ta.ema(df['close'], length=EMA_SLOW)
    
    # Volume confirmation
    df['avg_volume'] = ta.sma(df['volume'], length=VOLUME_PERIOD)
    df['volume_confirm'] = df['volume'] > (df['avg_volume'] * VOLUME_THRESHOLD)
    
    # Price momentum
    df['price_change'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['strong_momentum'] = np.abs(df['price_change']) > MOMENTUM_THRESHOLD
    
    # Trend strength using EMA distance
    df['ema_dist'] = (df['ema_fast'] - df['ema_slow']) / df['ema_slow'] * 100
    df['trend_strong'] = np.abs(df['ema_dist']) > TREND_STRENGTH_THRESHOLD
    
    # EMA trend conditions
    df['fast_above_slow'] = df['ema_fast'] > df['ema_slow']
    df['fast_below_slow'] = df['ema_fast'] < df['ema_slow']
    
    # Price position relative to EMAs
    df['price_above_both'] = (df['close'] > df['ema_fast']) & (df['close'] > df['ema_slow'])
    df['price_below_both'] = (df['close'] < df['ema_fast']) & (df['close'] < df['ema_slow'])
    
    # Entry conditions
    df['long_condition'] = (
        df['fast_above_slow'] & 
        df['price_above_both'] & 
        df['volume_confirm'] & 
        df['strong_momentum'] & 
        df['trend_strong']
    )
    df['short_condition'] = (
        df['fast_below_slow'] & 
        df['price_below_both'] & 
        df['volume_confirm'] & 
        df['strong_momentum'] & 
        df['trend_strong']
    )
    
    # Trigger signals only on transition (like crossover)
    #df['long_signal'] = df['long_condition'] & ~df['long_condition'].shift(1).fillna(False)
    #df['short_signal'] = df['short_condition'] & ~df['short_condition'].shift(1).fillna(False)
    
    # Trigger signals only on transition (like crossover)
    df['long_signal']  = df['long_condition']  & ~df['long_condition'].shift(1, fill_value=False)
    df['short_signal'] = df['short_condition'] & ~df['short_condition'].shift(1, fill_value=False)


    # Initialize signal arrays
    length = len(df)
    signals = np.zeros(length)
    buy_prices = np.zeros(length)
    stop_losses = np.zeros(length)
    take_profits = np.zeros(length)
    sides = [''] * length
    
    # Get decimal points for the price (for rounding)
    price_precision = 0
    for col in ['open', 'high', 'low', 'close']:
        if '.' in str(df[col].iloc[1]):
            precision = len(str(df[col].iloc[1]).split('.')[1])
            if precision > price_precision:
                price_precision = precision
    
    # Generate signals (adapted to prevent immediate re-entries, similar to original code)
    for i in range(1, length):
        if df['long_signal'].iloc[i] and sides[i-1] == '':
            signals[i] = 2  # Long signal
            sides[i] = 'Buy'
            buy_prices[i] = df['close'].iloc[i]
            stop_losses[i] = round(buy_prices[i] * (1 - RISK_PERCENT), price_precision)
            take_profits[i] = round(buy_prices[i] * (1 + RISK_PERCENT * REWARD_RATIO), price_precision)
        elif df['short_signal'].iloc[i] and sides[i-1] == '':
            signals[i] = 1  # Short signal
            sides[i] = 'Sell'
            buy_prices[i] = df['close'].iloc[i]
            stop_losses[i] = round(buy_prices[i] * (1 + RISK_PERCENT), price_precision)
            take_profits[i] = round(buy_prices[i] * (1 - RISK_PERCENT * REWARD_RATIO), price_precision)
    
    df['signal'] = signals
    df['side'] = sides
    df['buy_price'] = buy_prices
    df['sl'] = stop_losses
    df['tp'] = take_profits
    
    return df

def generate_trades_df(df):
    """
    Generate trades DataFrame with result and gain_percentage.
    
    Args:
        df: DataFrame with signals and OHLCV data
    
    Returns:
        trades_df: DataFrame with trade records
    """
    length = len(df)
    high = df['high'].values
    low = df['low'].values
    signal = df['signal'].values.copy()
    trades_list = []
    is_trade_open = False
    for line in range(length):
        if signal[line] == 0:
            is_trade_open = False
            continue
        
        trade_start_time = df.iloc[line]['time']
        buy_price = df.iloc[line]['buy_price']
        stop_loss = df.iloc[line]['sl']
        take_profit = df.iloc[line]['tp']
        side = df.iloc[line]['side']
        
        is_trade_open = True
        trade_closed = False
        trade_won = False
        trade_close_time = None
        gain_percentage = 0
        
        for i in range(1, length - line):
            current_idx = line + i
            if signal[current_idx] != 0:
                signal[current_idx] = 0  # Prevent overlapping trades
            
            if signal[line] == 1:  # Short signal
                if high[current_idx] >= stop_loss:
                    trade_won = False
                    trade_closed = True
                    trade_close_time = df.iloc[current_idx]['time']
                    gain_percentage = ((buy_price - stop_loss) / buy_price) * 100
                    is_trade_open = False
                    break
                elif low[current_idx] <= take_profit:
                    trade_won = True
                    trade_closed = True
                    trade_close_time = df.iloc[current_idx]['time']
                    gain_percentage = ((buy_price - take_profit) / buy_price) * 100
                    is_trade_open = False
                    break
            elif signal[line] == 2:  # Long signal
                if low[current_idx] <= stop_loss:
                    trade_won = False
                    trade_closed = True
                    trade_close_time = df.iloc[current_idx]['time']
                    gain_percentage = ((stop_loss - buy_price) / buy_price) * 100
                    is_trade_open = False
                    break
                elif high[current_idx] >= take_profit:
                    trade_won = True
                    trade_closed = True
                    trade_close_time = df.iloc[current_idx]['time']
                    gain_percentage = ((take_profit - buy_price) / buy_price) * 100
                    is_trade_open = False
                    break
        
        if trade_closed:
            trade_record = {
                'trade_start_time': trade_start_time,
                'trade_close_time': trade_close_time,
                'buy_price': buy_price,
                'tp': take_profit,
                'sl': stop_loss,
                'side': side,
                'result': 'win' if trade_won else 'lose',
                'gain_percentage': gain_percentage
                # 'is_virtual': False  # Default to False, to be determined in views.py
            }
            trades_list.append(trade_record)
        # Store the last not completed trade if it exists
        if not trade_closed and is_trade_open:
            trade_record = {
                'trade_start_time': trade_start_time,
                'trade_close_time': None,
                'buy_price': buy_price,
                'tp': take_profit,
                'sl': stop_loss,
                'side': side,
                'result': None,
                'gain_percentage': 0
            }
            trades_list.append(trade_record)
    
    trades_df = pd.DataFrame(trades_list)
    print(f"Generated trades DataFrame with {trades_list} trades")
    return trades_df

def process_incomplete_trade(last_trade, df_with_signals, coin_pair):
    """
    Process an incomplete trade by checking SL/TP and then process new trades.
    """
    print(f"process incomplete trade {coin_pair} last trade original time {last_trade.trade_start_time}")
    last_trade_start_time = pd.Timestamp(last_trade.trade_start_time)
    #print(f"process incomplete trade {coin_pair} last trade start time after pd.timestamp {last_trade_start_time}")
    if last_trade_start_time.tz is not None:
        last_trade_start_time = last_trade_start_time.tz_localize(None)
    #print(f"process incomplete trade {coin_pair} last trade start time after localize {last_trade_start_time}")
    #print(f"df with signals first trade time before localize {df_with_signals['time'].iloc[0]}")
    if df_with_signals["time"].dt.tz is not None:
        df_with_signals["time"] = df_with_signals["time"].dt.tz_localize(None)
    #print(f"df with signals first trade time after localize {df_with_signals['time'].iloc[0]}")
    df_after = df_with_signals[df_with_signals["time"] > last_trade_start_time].copy()
    #print(f"df_after first trade time after old signal removal {df_after['time'].iloc[0]} and last trade time is {last_trade_start_time}")
    if df_after.empty:
        print(f"No new data to process incomplete trade for {coin_pair}")
        return None

    buy_price = float(last_trade.buy_price)
    stop_loss = float(last_trade.sl)
    take_profit = float(last_trade.tp)
    side = last_trade.side

    trade_closed = False
    trade_won = False
    trade_close_time = None
    gain_percentage = 0

    for idx, row in df_after.iterrows():
        if side == 'Buy':
            if row['low'] <= stop_loss:
                trade_won = False
                trade_closed = True
                trade_close_time = row['time']
                gain_percentage = ((stop_loss - buy_price) / buy_price) * 100
                break
            elif row['high'] >= take_profit:
                trade_won = True
                trade_closed = True
                trade_close_time = row['time']
                gain_percentage = ((take_profit - buy_price) / buy_price) * 100
                break
        elif side == 'Sell':
            if row['high'] >= stop_loss:
                trade_won = False
                trade_closed = True
                trade_close_time = row['time']
                gain_percentage = ((buy_price - stop_loss) / buy_price) * 100
                break
            elif row['low'] <= take_profit:
                trade_won = True
                trade_closed = True
                trade_close_time = row['time']
                gain_percentage = ((buy_price - take_profit) / buy_price) * 100
                break
            
    if trade_closed:
        #print(f"process incomplete trade {coin_pair} last trade close time {trade_close_time}")
        last_trade.trade_close_time = trade_close_time
        last_trade.result = 'win' if trade_won else 'lose'
        last_trade.gain_percentage = gain_percentage
        last_trade.save()  # is_virtual determined in views.py
        print(f"Completed trade for {coin_pair} at {trade_close_time}")
        
        trade_close_time_ts = pd.Timestamp(trade_close_time)
        if trade_close_time_ts.tz is not None:
            trade_close_time_ts = trade_close_time_ts.tz_localize(None)
        
        df_after = df_with_signals[df_with_signals["time"] > trade_close_time_ts].copy()
        return df_after
    print(f"No new data to process incomplete trade for {coin_pair}")
    return None

def process_new_trades(df_with_signals, coin_pair):
    """
    Process new trades after the last completed trade, ensuring no overlap.
    """
    trades_df = generate_trades_df(df_with_signals)

    if trades_df.empty:
        #print(f"No new trades to process for {coin_pair}")
        return

    for _, trade in trades_df.iterrows():
        Trade.objects.create(
            coinpair_name=coin_pair,
            trade_start_time=trade['trade_start_time'],
            trade_close_time=trade['trade_close_time'],
            buy_price=trade['buy_price'],
            tp=trade['tp'],
            sl=trade['sl'],
            side=trade['side'],
            result=trade['result'],
            gain_percentage=trade['gain_percentage'],
            # is_virtual=False  # Default to False, to be determined in views.py
        )

    print(f"Saved {len(trades_df)} new trades for {coin_pair}")

def process_coin_pair(coin_pair_name, client):
    print(f"Processing {coin_pair_name}...")
    historical_data_1m = fetch_historical_data(client, coin_pair_name, '1m', limit=1000)
    #print(f"last candle for {coin_pair_name}: {historical_data_1m.iloc[-1] if historical_data_1m is not None else 'None'}")
    if historical_data_1m is None:
        print(f"Skipping {coin_pair_name} due to data fetch error")
        return

    historical_data_1m['symbol'] = coin_pair_name
    print(f"Fetched historical data for {coin_pair_name} with last rows")
    #print(historical_data_1m.tail(1))
    try:
        signals_df = generate_trading_signals(historical_data_1m)
        trades = Trade.objects.filter(coinpair_name=coin_pair_name).order_by('trade_start_time')
        
        if trades.exists():
            last_trade = trades.last()
            print(f"(Trade table) Last trade start time for {coin_pair_name}: {last_trade.trade_start_time}")
            #print(f"(Trade table) Last trade close time for {coin_pair_name}: {last_trade.trade_close_time} {type(last_trade.trade_close_time)}")
            if last_trade.trade_close_time is not None:
                print(f"(Trade table) Last trade for {coin_pair_name} is already closed, processing new trades...")
                last_trade_close_time = pd.Timestamp(last_trade.trade_close_time)
                if last_trade_close_time.tz is not None:
                    last_trade_close_time = last_trade_close_time.tz_localize(None)
                if signals_df["time"].dt.tz is not None:
                    signals_df["time"] = signals_df["time"].dt.tz_localize(None)
                signals_df = signals_df[signals_df["time"] > last_trade_close_time]
                if not signals_df.empty:
                    process_new_trades(signals_df, coin_pair_name)
            else:
                print(f"(Trade table) Processing incomplete trade for {coin_pair_name}...")
                signals_df = process_incomplete_trade(last_trade, signals_df, coin_pair_name)
                if signals_df is not None and not signals_df.empty:
                    process_new_trades(signals_df, coin_pair_name)
        else:
            print(f"No trades found for {coin_pair_name}, starting new backtest...")
            signals_df = signals_df.iloc[max(EMA_FAST, EMA_SLOW, VOLUME_PERIOD):]  # Skip initial rows for indicator warmup
            process_new_trades(signals_df, coin_pair_name)
    except Exception as e:
        print(f"(Trade table) Error processing {coin_pair_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    



    