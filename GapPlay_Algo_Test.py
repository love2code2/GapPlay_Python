from numpy import isnan, dot
import numpy as np
import pandas as pd
from scipy import stats
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline import CustomFactor
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.factors import RSI
from quantopian.pipeline.factors import AverageDollarVolume
from quantopian.pipeline.factors import SimpleMovingAverage
import statsmodels.api as sm

class MarketCap(CustomFactor):   
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close, morningstar.valuation.shares_outstanding] 
    window_length = 1    
    # Compute market cap value
    def compute(self, today, assets, out, close, shares):       
        out[:] = close[-1] * shares[-1]
        
class LastClose(CustomFactor):   
    # Pre-declare inputs and window_length
    inputs = [USEquityPricing.close] 
    window_length = 1
    # Compute last close
    def compute(self, today, assets, out, close): 
        out[:] = close[-1]
                
def initialize(context):   
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25, price_impact=0.1))
    set_commission(commission.PerShare(cost=0.000, min_trade_cost=0.00))    # Assume Robinhood zero-commissions
    
    # Filter order:
    # 1. MktCap > X # filters stocks with minimum size 
    # 2. Volume > X # filters stocks with minimum volume
    # 3. X < SMA_200 & X < SMA_10 # filter stocks with gaps more likely to fill   
    # 4. Y < RSI < X # filters stocks with (+) momentum and strong trend

    mkt_cap = MarketCap()
    last_close = LastClose()
    dollar_volume = AverageDollarVolume(window_length=10)
    sma_200 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=200)
    sma_10 = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=10)
    rsi = RSI(inputs=[USEquityPricing.close], window_length=15)

    mkt_cap_rank = mkt_cap.rank(ascending=False)
    mkt_cap_filter = (mkt_cap_rank < 4000)
    #sma_filter = (last_close > sma_10) # & (last_close < sma_200) # shouldnt be last close, should be open price
    dollar_volume_filter = (dollar_volume > 100000)
    rsi_filter = (rsi > 15) & (rsi < 50)

    pipe = Pipeline(
        columns={'mkt_cap_rank':mkt_cap_rank,
                 'last_close':last_close,
                 '$volume':dollar_volume,
                 'sma_200%':last_close/sma_200,
                 'sma_10%':last_close/sma_10,
                 'rsi/50':rsi/50
                },
        screen=(mkt_cap_filter & dollar_volume_filter & rsi_filter) # & sma_filter)
    )
    attach_pipeline(pipe, name='my_pipeline')
    
    for i in range(2,5):
        schedule_function( # This will execute buys at 9:31AM
        openFunction,
        date_rules.week_start(days_offset=i), #do not run on Mondays (poor success)
        time_rules.market_open(minutes=1),
        False #exclude half days
        )    
        schedule_function( # This will sell open positions before close
        closeFunction,
        date_rules.week_start(days_offset=i), #do not run on Mondays (poor success)
        time_rules.market_close(minutes=30),
        False #exclude half days
        )    

def openFunction(context, data): 
    keys = range(len(context.stocks)) # keys for eventual stocks to buy
    
    # Linear regression to narrow stocks to those with an uptrend, remove undesirables from keys
    prices = data.history(context.stocks, 'open', 252/2, '1d') #252 is 1yr
    X=range(len(prices))
    A=sm.add_constant(X) # Add column of ones so we get intercept
    for i in range(len(context.stocks)):
        stock = context.stocks[i]
        sd = prices[stock].std() # Price movement
        Y = prices[stock].values # Price points to run regression
        if isnan(Y).any(): # If all empty, skip
            continue
        results = sm.OLS(Y,A).fit() # Run regression y = ax + b
        (b, a) =results.params
        slope = a / b * 252   # Normalized slope     # Daily return regression * 1yr
        #delta = Y - (dot(a,X) + b) # Currently how far away from regression line?
        slope_min = 0.252  # Don't trade if the slope is near flat, < 7%/yr
        if slope < slope_min:
            if keys.count(i)>0:            
                continue
                keys.remove(i) # remove undesirables from keys
                
    # Remove stocks with relatively high volume (which may signal a trend reversal)
    #volumes_24h = history(1441,'1m','volume',ffill=False).as_matrix(data)[:,:]
    #for i in range(len(context.stocks)):
    #    stock = context.stocks[i]
    #    yday_open_volume = volumes_24h[0,i] 
    #    if data[stock].volume > yday_open_volume * 3: # relative high volume compared to yday
    #        if keys.count(i)>0:
    #            continue
    #            keys.remove(i) # remove undesirables from keys   
                
    highs_4d = data.history(context.stocks, 'high', 5, '1d').as_matrix(None)[:-1,:]
    lows_4d = data.history(context.stocks, 'low', 5, '1d').as_matrix(None)[:-1,:]     

    # Calculate gap sizes to narrow stocks to those with gaps, remove undesirables from keys
    opens = data.history(context.stocks, 'open', 31, '1d').as_matrix(None)[1:,:]
    closes = data.history(context.stocks, 'close', 31, '1d').as_matrix(None)[:-1,:]
    gaps = closes - opens
    gaps_z = stats.zscore(gaps, axis=0, ddof=1)    # not currently used but could be to normalize
    for i in range (len(gaps)):
        if gaps[-1,i] > 0: # only down gaps allowed since we can't short 
            gap_size = gaps[-1,i] / closes[-1,i]
            if gap_size < 0.005 or gap_size > 0.025: # not too big, not too small
                if keys.count(i)>0:
                    keys.remove(i)            
        else: # remove up gaps since we can't short sell
            if keys.count(i)>0:
                keys.remove(i) # remove undesirables from keys            
                
    # Determine if the open for a gap is between the past 4d high/low and remove these undesirables
    for i in range(len(context.stocks)):
        high = max(highs_4d[:,i]) # highest high in past 4d
        low = min(lows_4d[:,i]) # lowest low in past 4d    
        if opens[-1,i] < high and opens[-1,i] > low:
            if keys.count(i)>0:
                keys.remove(i) # remove undesirables from keys
                
    num_tradeable_gaps = len(keys)
    print num_tradeable_gaps
    if num_tradeable_gaps > 0:
        order_percentage = np.minimum(0.3, (100 / num_tradeable_gaps) * 0.01) # max to invest in any 1 stock is 30%
        for k in range(num_tradeable_gaps):
            key = keys[k]
            stock = context.stocks[key]
            log.info("{stock} --> Gap Size: {gap_size}, Z-score: {z_score}".format(stock=stock, gap_size=gaps[-1,key], z_score=gaps_z[-1,key]))            
            if context.portfolio.cash > order_percentage * context.portfolio.portfolio_value:
                if not get_open_orders(stock):
                    order_target_percent(stock, order_percentage)
    
def closeFunction(context, data):
    opens = data.history(context.stocks, 'open', 2,'1d').as_matrix(None)[1,:] 
    
    for i in range(len(context.stocks)):
        stock = context.stocks[i]
        amount_held = context.portfolio.positions[stock].amount
        if amount_held > 0:
            if not get_open_orders(stock):
                profit = np.around(((data.current(stock, 'price') - opens[i]) / opens[i] * 100.00), decimals=3)
                if profit >= 0:
                    log.info("MARKET CLOSE {stock}->PROFIT:{profit} %".format(stock=stock, profit=profit))
                else:
                    log.info("MARKET CLOSE {stock}->LOSS:{profit} %".format(stock=stock, profit=profit)) 
                order_target_percent(stock, 0)
    
def before_trading_start(context, data):
    context.stocks = pipeline_output('my_pipeline').sort('mkt_cap_rank').index[:100].T

def handle_data(context, data):
    record(cash=context.portfolio.cash/context.portfolio.starting_cash)
    record(holdings=context.portfolio.positions_value/context.portfolio.starting_cash)
    record(leverage=context.account.leverage)
    
    # Target met -> Sell
    closes = data.history(context.stocks, 'close', 2, '1d').as_matrix(None)[0,:]    
    opens = data.history(context.stocks, 'open', 2,'1d').as_matrix(None)[1,:]    
    for i in range(len(context.stocks)):
        stock = context.stocks[i]
        position = context.portfolio.positions[stock]
        if position.amount > 0: 
            gap_size = closes[i] - opens[i]
            if data.current(stock, 'price') >= (opens[i] + (gap_size * 1.0)): # fill gap 80% for target
                if not get_open_orders(stock):                    
                    profit = np.around(((data.current(stock, 'price') - opens[i]) / opens[i] * 100.00), decimals=3)
                    log.info("TARGET MET {stock}->PROFIT:{profit} %".format(stock=stock, profit=profit))
                    order_target_percent(stock, 0)
    
    # Stop Loss
    for i in range(len(context.stocks)):
        stock = context.stocks[i]
        position = context.portfolio.positions[stock]
        if position.amount > 0:
            if data.current(stock, 'price') < opens[i] * 0.98:
                if not get_open_orders(stock):
                    loss = np.around(((data.current(stock, 'price') - opens[i]) / opens[i] * 100.00), decimals=3)
                    log.info("STOP LOSS {stock}->LOSS:{loss} %".format(stock=stock, loss=loss))              
                    order_target_percent(stock, 0)