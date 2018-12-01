# python version: 3.6.5
# matplotlib version: 2.2.2
# numpy version: 1.14.3
# pandas version: 0.23.0
# Author: Xin Zhou
import quandl
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
from copy import copy

#-----------------------------------------
# get data
quandl.ApiConfig.api_key = "wj_waHihRn_8tZaKXxEb"
def create_unadjusted_single_month(front_month_contract, back_month_contract):
    # Return unadjusted active contract for the front month using the front month contract data and back month contract data
    # For example, for contract 2018-04 and 2018-05, suppose the roll date is 2018-04-15, then this function will return a new 1 month(2018-04) contract
    # with 2018-04-01 to 2018-04-15 data from contract 2018-04 and 2018-04-15 to 2018-04-30 data from contract 2018-05. 
    # Besides, it also return the close price gap between front month contract and back month contract on roll date

    front_month_expire_date = front_month_contract.index[-1]
    expire_month = front_month_expire_date.strftime('%Y-%m')

    df_front_active = front_month_contract[expire_month]
    df_back_active = back_month_contract[expire_month]

    # calculate roll date based on volume
    combined_df_volume = pd.concat([df_front_active.Volume, df_back_active.Volume], axis=1)
    
    if len(combined_df_volume[combined_df_volume.iloc[:, 0] < combined_df_volume.iloc[:, 1]]) == 0:
        roll_date = df_front_active.index[-1]
    else:
        roll_date = combined_df_volume[combined_df_volume.iloc[:, 0] < combined_df_volume.iloc[:, 1]].index[0]

    df_front_active = df_front_active[:roll_date]
    df_back_active = df_back_active[roll_date:]
    gap = df_back_active.Close.iloc[0] - df_front_active.Close.iloc[-1] # price gap between front month contract and back month contract on roll date
    df_front_active = df_front_active.iloc[:-1, :]
    df_active = pd.concat([df_front_active, df_back_active], axis=0)[['Open', 'High', 'Low', 'Close']]
    return df_active, roll_date, gap
def get_adjusted_data(year_list):
    # Return adjusted continuous contract for given years
    # Backward panama canal method is used to get continuous contract

    # ----------------------------------------------------------------
    # Step 1. Get the raw historical data for each contract stored in an ordered Dict using format 'month_year' as keys.
    month_code = collections.OrderedDict({'01': 'ICF',
                '02': 'ICG',
                '03': 'ICH',
                '04': 'ICJ',
                '05': 'ICK',
                '06': 'ICM',
                '07': 'ICN',
                '08': 'ICQ',
                '09': 'ICU',
                '10': 'ICV',
                '11': 'ICX',
                '12': 'ICZ'})
    raw_df_dict = collections.OrderedDict()
    num_tickers = len(year_list)*len(month_code)
    print('Num of tickers to be retrieved: ', num_tickers)
    count = 0
    for year in year_list:
        for month in month_code.keys():
            ticker = "CFFEX/" + month_code[month]+year
            name = month+'_'+year
            df = quandl.get(ticker)[['Open', 'High', 'Low', 'Close', 'Volume']] 
            raw_df_dict[name] = df

            count += 1
            print(count) # Diplay how many tickers we have downloaded so far, can be removed later.
    # Step2. Create a coninuous unadjusted contract with roll date defined as the date when the back month contract volume is 
    # larger than the front month contract volume for the first time.
    df_unadjusted = pd.DataFrame()
    roll_date_list = [] # store the roll date of each month used later for creating continuous contract
    gap_list = [] # store the close price gap at the roll date between front month contract and back month contract, will be used later for creating continuous contract.
    for ii in range(len(list(raw_df_dict.keys()))-1):   
        front_month = list(raw_df_dict.keys())[ii]
        back_month = list(raw_df_dict.keys())[ii+1]
        df_cur, roll_date, gap = create_unadjusted_single_month(raw_df_dict[front_month], raw_df_dict[back_month])
        df_unadjusted = pd.concat([df_unadjusted, df_cur], axis=0)
        roll_date_list.append(roll_date)
        gap_list.append(gap)
    #  Step3. Create back adjusted continuous contract. 
    df_adjusted = df_unadjusted.copy()
    for jj in range(len(roll_date_list)):
        roll_date_index = df_adjusted.index.tolist().index(roll_date_list[jj])
        prev_roll_date_index = df_adjusted.index.tolist().index(roll_date_list[jj-1])
        if jj == 0:
            df_adjusted.iloc[:roll_date_index]  += sum(gap_list) 
        else:
            df_adjusted.iloc[prev_roll_date_index:roll_date_index] += sum(gap_list[jj:])

    return df_adjusted

#-----------------------------------------
# indicators
def MA_indicator(close, period):
    MA_series = close.rolling(window=period).mean()
    return MA_series
def RSI_indicator(close, period):
    df = pd.DataFrame(close)
    df['U'] = df['Close'] - df['Close'].shift(1)
    df['U'][df['U'] < 0] = 0.0
    df['D'] = df['Close'] - df['Close'].shift(1)
    df['D'][df['D'] > 0] = 0.0
    df['D'] = abs(df['D'])
    
    df['first_avg_U'] = df['U'].rolling(window=period).mean()
    df['first_avg_D'] = df['D'].rolling(window=period).mean()
    
    df = df.dropna()
    
    df['U'].iloc[0] = df['first_avg_U'].iloc[0]
    df['D'].iloc[0] = df['first_avg_D'].iloc[0]
    
    df['avg_U'] = EMA_indicator(df['U'], period)
    df['avg_D'] = EMA_indicator(df['D'], period)
    
    df['RS'] = df['avg_U']/df['avg_D']
    df['RSI'] = 100.0 - 100.0/(1+df['RS'])
    
    return df['RSI']
def EMA_indicator(close, period):
    EMA_series = close.ewm(com=period-1, adjust=False).mean()
    return EMA_series

# ----------------------------------------
# backtest metrics
def backtest_metrics(df_daily_pnl, df_trade, trade_pnl, strategy_type):
    # Return a dictionary contains backtesting metrics like sharpe ratio, drawdown.
    d = {}
    #d['initial_fund'] = df_daily_pnl['net_PnL'].iloc[0]
    #d['final_fund'] = df_daily_pnl['net_PnL'].iloc[-1]
    d['start_date'] = df_daily_pnl['date'].iloc[0]
    d['end_date'] = df_daily_pnl['date'].iloc[-1]
    d['total_return'] = round(df_daily_pnl['net_PnL'].iloc[-1], 2)
    d['total_trade_counts'] = len(trade_pnl)
    d['win_trade_counts'] = len(trade_pnl[trade_pnl['profit'] > 0.0])
    d['loss_trade_counts'] = len(trade_pnl[trade_pnl['profit'] < 0.0])
    d['avg_win_per_trade'] = round(trade_pnl[trade_pnl['profit'] > 0.0]['profit'].mean(), 2)
    d['avg_loss_per_trade'] = round(trade_pnl[trade_pnl['profit'] < 0.0]['profit'].mean(), 2)    
    d['max_drawdown'] = round(df_daily_pnl['daily_drawdown'].min(), 2)
    d['sharpe_ratio'] = round(np.sqrt(252) * df_daily_pnl['daily_pnl'].mean() / df_daily_pnl['daily_pnl'].std(), 2)
    print('Backtesting results for '+strategy_type)
    print('------------------------------------------')
    for ii in d.keys():
        print(ii, ":", d[ii])
    print(' ')
    print(' ')
    return d

# plot backtesting results
def plot_results_matplotlib(price, df_trade, df_daily_pnl, trade_pnl, strategy_type):


    plt.figure(figsize=(20, 10))

    plt.subplot(5, 1, 1)
    plt.title(strategy_type + '\n' + 'Cumulative PnL')
    plt.plot(df_daily_pnl['date'], df_daily_pnl['net_PnL'])
   
    plt.subplot(5, 1, 2)
    plt.title('Daily position')
    plt.plot(df_daily_pnl['date'].values, df_daily_pnl.daily_pos)
    
    plt.subplot(5, 1, 3)
    plt.title('Price')
    plt.plot(price['Close'], label='Close Price')
    plt.plot(trade_pnl[trade_pnl['enter_direction'] == 'Long']['enter_date'], trade_pnl[trade_pnl['enter_direction'] == 'Long']['enter_price'], 'm.', markersize=10, label='Long')
    plt.plot(trade_pnl[trade_pnl['enter_direction'] == 'Short']['enter_date'], trade_pnl[trade_pnl['enter_direction'] == 'Short']['enter_price'], 'y.', markersize=10, label='Short')
    plt.plot(trade_pnl['exit_date'], trade_pnl['exit_price'], 'k.', markersize=10, label='Cover')
    plt.legend()

    plt.subplot(5, 1, 4)
    plt.title('Drawdown')
    plt.fill_between(df_daily_pnl['date'].values, df_daily_pnl['daily_drawdown'])
    plt.ylim([df_daily_pnl['daily_drawdown'].min(), 0.0])
    
    plt.subplot(5, 1, 5)
    plt.title('Trade PnL distribution')
    plt.hist(trade_pnl['profit'], bins=50, align='mid')

    plt.tight_layout()
    plt.show()

# ----------------------------------------
# backtesting
# momentum
def run_backtest_momentum(price, parameters):
    # Return four dataframes after backtesting: price, df_trade, df_daily_pnl, trade_pnl
    # price: contains close price and indicator values associated with the strategy
    # df_trade: each row contains one trade information(date, price, quantity, direction)
    # df_daily_pnl: contains mark-to-market daily PnL, daily position, daily drawdown
    # trade_pnl: each row contains the long or short trade with its corresponding cover trade, 
    # the last column is the profit of these two trades.


    # backtesting variables
    trade_list = [] # store trade
    trade_date_list = [] # store trade date
    trade_enter = {} # store the information(price, date, quantity, direction) of enter trade
    trade_exit = {} # store the information(price, date, quantity, direction)  of exit trade
    daily_pnl_list = [] # store the daily pnl
    daily_pnl_date_list = [] # store the corresponding date for daily pnl
    daily_pos_list = [] # store the position on each day

    # strategy variables
    initial_fund_const = parameters['initial_fund'] # this is is the initial fund we have
    initial_fund = copy(initial_fund_const) # this one will varies as we make profit or loss
    trailing_ratio = parameters['trailing_ratio']  # used for trailing stop loss

    MA_fast_window = parameters['MA_fast_window']
    MA_slow_window = parameters['MA_slow_window']   
    MA_fast = MA_indicator(price['Close'].copy(), MA_fast_window)
    MA_slow = MA_indicator(price['Close'].copy(), MA_slow_window)
    MA_fast.name = 'MA_fast'
    MA_slow.name = 'MA_slow'
    
    price = pd.concat([price, MA_fast, MA_slow], axis=1)
    price = price.dropna()
    pos = False # check if we have position or not
    
    # start backtesting
    # there are 6 kinds of situations may happen during each loop in backtesting
    # 1. long signal, enter long position
    # 2. short signal, enter short position
    # 3. cover long position
    # 4. cover short position
    # 5. no long or short signal, but has position
    # 6. no long or short signal, no position
    #
    # We use yesterday's information to generate signal(exter or sell signal), then use today's price to execute trade.
    # This is to avoid lookahead bias.
    for ii in range(1, len(price)): 
        cur_open = price['Open'].iloc[ii]
        cur_high = price['High'].iloc[ii]
        cur_low = price['Low'].iloc[ii]
        cur_close = price['Close'].iloc[ii]
        cur_date = price.index[ii]

        prev_open = price['Open'].iloc[ii-1]
        prev_high = price['High'].iloc[ii-1]
        prev_low = price['Low'].iloc[ii-1]
        prev_close = price['Close'].iloc[ii-1]
        prev_date = price.index[ii-1]

        prev_MA_fast = price['MA_fast'].iloc[ii-1]
        prev_MA_slow = price['MA_slow'].iloc[ii-1]
    
        #1. long signal, enter long position
        if prev_MA_fast > prev_MA_slow and pos == False:
            intraTradeHigh = cur_close # this is used for a trailing stop loss.
            # store information of long trade
            trade_enter['date'] = cur_date
            trade_enter['price'] = cur_close
            trade_enter['quantity'] = 1.0 
            trade_enter['direction'] = 'Long'
            
            trade_list.append(trade_enter)
            trade_date_list.append(cur_date)
            
            daily_pnl_list.append(0.0)
            daily_pnl_date_list.append(cur_date)
            daily_pos_list.append(1.0)

            # set the position status to be True
            pos = True
            continue
        #2. short signal, enter short position
        if prev_MA_fast < prev_MA_slow and pos == False:
            intraTradeLow = cur_close # this is used for a trailing stop loss.
            # store information of short trade
            trade_enter['date'] = cur_date
            trade_enter['price'] = cur_close
            trade_enter['quantity'] = 1.0 
            trade_enter['direction'] = 'Short'
            
            trade_list.append(trade_enter)
            trade_date_list.append(cur_date)
            
            daily_pnl_list.append(0.0)
            daily_pnl_date_list.append(cur_date)
            daily_pos_list.append(-1.0)
            
            # set the position status to be True
            pos = True
            continue
        # 3. cover long position
        if pos == True and trade_enter['direction'] == 'Long':
            intraTradeHigh = max(prev_high, intraTradeHigh)
            if prev_high < intraTradeHigh*(1 - trailing_ratio):
                # we first keep track of daily pnl
                daily_pnl_list.append(trade_enter['quantity']*(cur_close-prev_close))
                daily_pnl_date_list.append(cur_date)
                daily_pos_list.append(0.0)
                initial_fund = initial_fund_const + sum(daily_pnl_list) # this is the total fund we have after closing the position
                
                # store information of exit trade
                trade_exit['date'] = cur_date
                trade_exit['price'] = cur_close
                trade_exit['quantity'] = trade_enter['quantity']
                trade_exit['direction'] = 'Cover_Long'
                trade_list.append(trade_exit)
                trade_date_list.append(cur_date)
                
                # set the position status to be False
                pos = False
                trade_enter = {}
                trade_exit = {}
                continue
        # 4. cover short position
        if pos == True and trade_enter['direction'] == 'Short':
            intraTradeLow = min(prev_low, intraTradeLow)
            if prev_low > intraTradeLow*(1 + trailing_ratio):
                # we first keep track of daily pnl
                daily_pnl_list.append(trade_enter['quantity']*(prev_close-cur_close))
                daily_pnl_date_list.append(cur_date)
                daily_pos_list.append(0.0)
                initial_fund = initial_fund_const + sum(daily_pnl_list) # this is the total fund we have after closing the position
                
                # store information of exit trade
                trade_exit['date'] = cur_date
                trade_exit['price'] = cur_close
                trade_exit['quantity'] = trade_enter['quantity']
                trade_exit['direction'] = 'Cover_Short'
                trade_list.append(trade_exit)
                trade_date_list.append(cur_date)
                
                # set the position status to be False
                pos = False
                trade_enter = {}
                trade_exit = {}
                continue

        # keep track of pnl when no trade happen
        # 5. no long or short signal, but has position
        if pos == True:
            if cur_date not in daily_pnl_date_list:
                if trade_enter['direction'] == 'Long':
                    cur_pnl = (cur_close-prev_close)*trade_enter['quantity']
                    daily_pnl_list.append(cur_pnl)
                    daily_pnl_date_list.append(cur_date)
                    daily_pos_list.append(1.0)
    
                if trade_enter['direction'] == 'Short':
                    cur_pnl = (prev_close-cur_close)*trade_enter['quantity']
                    daily_pnl_list.append(cur_pnl)
                    daily_pnl_date_list.append(cur_date)
                    daily_pos_list.append(-1.0)
        #6. no long or short signal, no position
        elif pos == False:
            if cur_date not in daily_pnl_date_list:
                daily_pnl_list.append(0.0)
                daily_pnl_date_list.append(cur_date)
                daily_pos_list.append(0.0)


    df_trade = pd.DataFrame(trade_list)
    df_daily_pnl = pd.DataFrame({'date': daily_pnl_date_list, 'daily_pnl': daily_pnl_list, 'daily_pos': daily_pos_list})
    df_daily_pnl['net_PnL'] = df_daily_pnl['daily_pnl'].cumsum() + initial_fund_const
    df_daily_pnl['daily_drawdown'] = [-max(0.0, (df_daily_pnl['net_PnL'].iloc[:(ii+1)].max() - df_daily_pnl['net_PnL'].iloc[ii])) for ii in range(len(df_daily_pnl))]

    # calculate profit for each trade
    enter_trade = df_trade.iloc[::2]
    exit_trade = df_trade.iloc[1::2]
    enter_trade.columns = ['enter_'+ii for ii in enter_trade.columns]
    exit_trade.columns = ['exit_'+ii for ii in exit_trade.columns]
    enter_trade = enter_trade.reset_index().drop(['index'], axis=1)
    exit_trade = exit_trade.reset_index().drop(['index'], axis=1)

    trade_pnl = pd.concat([enter_trade, exit_trade], axis=1)
    trade_pnl = trade_pnl.dropna() # in case there is a uncovered position at the end of backtesting, we drop it

    conditions = [trade_pnl['enter_direction'] == 'Long', trade_pnl['enter_direction'] == 'Short']
    choice = [1., -1.]
    trade_pnl['profit'] = (trade_pnl['exit_price']-trade_pnl['enter_price'])*np.select(conditions, choice)*trade_pnl['exit_quantity']

    return price, df_trade, df_daily_pnl, trade_pnl

# ----------------------------------------
# backtesting
# mean reversal
def run_backtest_mean_reversal(price, parameters):  
    # Return four dataframes after backtesting: price, df_trade, df_daily_pnl, trade_pnl
    # price: contains close price and indicator values assotiated with the strategy
    # df_trade: each row contains one trade information(date, price, quantity, direction)
    # df_daily_pnl: contains mark-to-market daily PnL, daily position, daily drawdown
    # trade_pnl: each row contains the long or short trade with its corresponding cover trade, 
    # the last column is the profit of these two trades.


    #backtesting variables
    trade_list = [] # store trade
    trade_date_list = [] # store trade date
    trade_enter = {} # store the information(price, date, quantity, direction) of enter trade
    trade_exit = {} # store the information(price, date, quantity, direction) of exit trade
    daily_pnl_list = [] # store the daily pnl
    daily_pnl_date_list = [] # store the corresponding date for daily pnl
    daily_pos_list = [] # store the position held each day

    # strategy variables
    initial_fund_const = parameters['initial_fund'] # this is is the initial fund we have
    initial_fund = copy(initial_fund_const) # this one will varies as we make profit or loss
    trailing_ratio = parameters['trailing_ratio'] # used for trailing stop loss
    
    RSI_period = parameters['RSI_period']
    RSI_threshold_upper = parameters['RSI_threshold_upper']
    RSI_threshold_lower = parameters['RSI_threshold_lower']
    
    RSI = RSI_indicator(price['Close'].copy(), RSI_period)
    price = pd.concat([price, RSI], axis=1)
    price = price.dropna()
    pos = False # check if we have position or not
    
    # start backtesting
    # there are 6 kinds of situations may happen during each loop in backtesting
    # 1. long signal, enter long position
    # 2. short signal, enter short position
    # 3. cover long position
    # 4. cover short position
    # 5. no long or short signal, but has position
    # 6. no long or short signal, no position
    #
    # We use yesterday's information to generate signal(exter or sell signal), then use today's price to execute trade.
    # this is to avoid lookahead bias.

    for ii in range(1, len(price)): # we start from 1 instead of 0 to avoid lookahead bias when enter the market. Will explain in detail later
        cur_open = price['Open'].iloc[ii]
        cur_high = price['High'].iloc[ii]
        cur_low = price['Low'].iloc[ii]
        cur_close = price['Close'].iloc[ii]
        cur_date = price.index[ii]

        prev_open = price['Open'].iloc[ii-1]
        prev_high = price['High'].iloc[ii-1]
        prev_low = price['Low'].iloc[ii-1]
        prev_close = price['Close'].iloc[ii-1]
        prev_date = price.index[ii-1]

        prev_RSI = price['RSI'].iloc[ii-1]
        pprev_RSI = price['RSI'].iloc[ii-2]
    
        #1. long signal, enter long position
        if prev_RSI < RSI_threshold_lower and pos == False:
            intraTradeHigh = cur_close # this is used for a trailing stop loss.
            # store information of long trade
            trade_enter['date'] = cur_date
            trade_enter['price'] = cur_close
            trade_enter['quantity'] = 1.0 #initial_fund/cur_close
            trade_enter['direction'] = 'Long'
            
            trade_list.append(trade_enter)
            trade_date_list.append(cur_date)
            
            daily_pnl_list.append(0.0)
            daily_pnl_date_list.append(cur_date)
            daily_pos_list.append(1.0)

            # set the position status to be True
            pos = True
            continue
        # 2. short signal, enter short position
        if prev_RSI > RSI_threshold_upper and pos == False:
            intraTradeLow = cur_close # this is used for a trailing stop loss.
            # store information of short trade
            trade_enter['date'] = cur_date
            trade_enter['price'] = cur_close
            trade_enter['quantity'] = 1.0 #initial_fund/cur_close
            trade_enter['direction'] = 'Short'
            
            trade_list.append(trade_enter)
            trade_date_list.append(cur_date)
            
            daily_pnl_list.append(0.0)
            daily_pnl_date_list.append(cur_date)
            daily_pos_list.append(-1.0)

            # set the position status to be True
            pos = True
            continue
        # 3. cover long position
        if pos == True and trade_enter['direction'] == 'Long':
            intraTradeHigh = max(prev_high, intraTradeHigh)
            if prev_high < intraTradeHigh*(1 - trailing_ratio):
                # we first keep track of daily pnl
                daily_pnl_list.append(trade_enter['quantity']*(cur_close-prev_close))
                daily_pnl_date_list.append(cur_date)
                daily_pos_list.append(0.0)
                initial_fund = initial_fund_const + sum(daily_pnl_list) # this is the total fund we have after closing the position
                
                # store information of exit trade
                trade_exit['date'] = cur_date
                trade_exit['price'] = cur_close
                trade_exit['quantity'] = trade_enter['quantity']
                trade_exit['direction'] = 'Cover_Long'
                trade_list.append(trade_exit)
                trade_date_list.append(cur_date)
                
                # set the position status to be False
                pos = False
                trade_enter = {}
                trade_exit = {}
                continue
        # 3. cover short position
        if pos == True and trade_enter['direction'] == 'Short':
            intraTradeLow = min(prev_low, intraTradeLow)
            if prev_low > intraTradeLow*(1 + trailing_ratio):
                # we first keep track of daily pnl
                daily_pnl_list.append(trade_enter['quantity']*(prev_close-cur_close))
                daily_pnl_date_list.append(cur_date)
                daily_pos_list.append(0.0)
                initial_fund = initial_fund_const + sum(daily_pnl_list) # this is the total fund we have after closing the position
                
                # store information of exit trade
                trade_exit['date'] = cur_date
                trade_exit['price'] = cur_close
                trade_exit['quantity'] = trade_enter['quantity']
                trade_exit['direction'] = 'Cover_Short'
                trade_list.append(trade_exit)
                trade_date_list.append(cur_date)
                
                # set the position status to be False
                pos = False
                trade_enter = {}
                trade_exit = {}
                continue
        # keep track of pnl when no trade happen
        # 5. no long or short signal, but has position
        if pos == True:
            if cur_date not in daily_pnl_date_list:
                if trade_enter['direction'] == 'Long':
                    cur_pnl = (cur_close-prev_close)*trade_enter['quantity']
                    daily_pnl_list.append(cur_pnl)
                    daily_pnl_date_list.append(cur_date)
                    daily_pos_list.append(1.0)
                if trade_enter['direction'] == 'Short':
                    cur_pnl = (prev_close-cur_close)*trade_enter['quantity']
                    daily_pnl_list.append(cur_pnl)
                    daily_pnl_date_list.append(cur_date)
                    daily_pos_list.append(-1.0)
        # 6. no long or short signal, no position
        elif pos == False:
            if cur_date not in daily_pnl_date_list:
                daily_pnl_list.append(0.0)
                daily_pnl_date_list.append(cur_date)
                daily_pos_list.append(0.0)

    df_trade = pd.DataFrame(trade_list)
    df_daily_pnl = pd.DataFrame({'date': daily_pnl_date_list, 'daily_pnl': daily_pnl_list, 'daily_pos': daily_pos_list})
    df_daily_pnl['net_PnL'] = df_daily_pnl['daily_pnl'].cumsum() + initial_fund_const
    df_daily_pnl['daily_drawdown'] = [-max(0.0, (df_daily_pnl['net_PnL'].iloc[:(ii+1)].max() - df_daily_pnl['net_PnL'].iloc[ii])) for ii in range(len(df_daily_pnl))]

    # calculate profit for each trade
    enter_trade = df_trade.iloc[::2]
    exit_trade = df_trade.iloc[1::2]
    enter_trade.columns = ['enter_'+ii for ii in enter_trade.columns]
    exit_trade.columns = ['exit_'+ii for ii in exit_trade.columns]
    enter_trade = enter_trade.reset_index().drop(['index'], axis=1)
    exit_trade = exit_trade.reset_index().drop(['index'], axis=1)

    trade_pnl = pd.concat([enter_trade, exit_trade], axis=1)
    trade_pnl = trade_pnl.dropna() # in case there is a uncovered position at the end of backtesting, we drop it

    conditions = [trade_pnl['enter_direction'] == 'Long', trade_pnl['enter_direction'] == 'Short']
    choice = [1., -1.]
    trade_pnl['profit'] = (trade_pnl['exit_price']-trade_pnl['enter_price'])*np.select(conditions, choice)*trade_pnl['exit_quantity']

    return price, df_trade, df_daily_pnl, trade_pnl

# ----------------------------------------
if __name__ == "__main__":
    print('Start downloading data from Quandl')
    print('Takes about 20s .................')
    year_list = ['2017', '2018']
    df_adjusted =  get_adjusted_data(year_list)
    print('Finish downloading')
    print('  ')
    print('  ')

    # running momentum strategy
    parameters_momentum = {'initial_fund': 0.0,
                'trailing_ratio': 0.01,
                'MA_fast_window': 50,
                'MA_slow_window': 200
    }
    
    price_momentum, df_trade_momentum, df_daily_pnl_momentum, trade_pnl_momentum = run_backtest_momentum(df_adjusted.copy(), parameters_momentum)
    plot_results_matplotlib(price_momentum, df_trade_momentum, df_daily_pnl_momentum, trade_pnl_momentum, strategy_type='Momentum Strategy')
    backtest_metrics_momentum = backtest_metrics(df_daily_pnl_momentum, df_trade_momentum, trade_pnl_momentum, strategy_type='Momentum Strategy')
    trade_pnl_momentum.to_csv('pnl_momentum.csv')
    


    parameters_mean_reversal = {'initial_fund': 0.0,
              'trailing_ratio': 0.001,
              'RSI_period': 2,
              'RSI_threshold_upper': 90,
              'RSI_threshold_lower': 10
    }
    price_mean_reversal, df_trade_mean_reversal, df_daily_pnl_mean_reversal, trade_pnl_mean_reversal = run_backtest_mean_reversal(df_adjusted.copy(), parameters_mean_reversal)
    plot_results_matplotlib(price_mean_reversal, df_trade_mean_reversal, df_daily_pnl_mean_reversal, trade_pnl_mean_reversal, strategy_type='Mean Reversal Strategy')
    backtest_metrics_mean_reversal = backtest_metrics(df_daily_pnl_mean_reversal, df_trade_mean_reversal, trade_pnl_mean_reversal, strategy_type='Mean Reversal Strategy')
    trade_pnl_mean_reversal.to_csv('pnl_mean_reversal.csv')

