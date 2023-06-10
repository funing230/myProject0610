import math

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mealpy.evolutionary_based import GA
from mealpy.swarm_based import GWO
from permetrics.regression import RegressionMetric
import tensorflow as tf
import statsmodels.regression.linear_model as rg
import numpy as np
import random
random.seed(7)
np.random.seed(42)
tf.random.set_seed(116)
from GA_util_all_data import print_table,pdmdd,normalize_series,triple_barrier,calculate_mdd,get_mdd
from numpy import array, reshape
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from baseline_util import get_z_socre_hege,get_z_socre_no_hege,get_z_socre_two_windows
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from result_util import get_pairstrategy_return


def get_mdd(x):
    """
    MDD(Maximum Draw-Down)
    :return: (mdd rate)
    """
    arr_v = np.array(x)
    peak_lower = np.argmax(np.maximum.accumulate(arr_v) - arr_v)
    peak_upper = np.argmax(arr_v[:peak_lower])
    return (arr_v[peak_lower] - arr_v[peak_upper]) / arr_v[peak_upper]

def get_my_mdd(returns):
    cumulative_returns = [1]
    for i in range(len(returns)):
        cumulative_return = cumulative_returns[i] * (1 + returns[i])
        cumulative_returns.append(cumulative_return)

    peak = max(cumulative_returns)
    valley = min(cumulative_returns[cumulative_returns.index(peak):])
    max_drawdown = (valley - peak) / peak
    return max_drawdown
# Define the function to print the table
def print_table(table):
    # Calculate the width of each column
    col_width = [max(len(str(row[i])) for row in table) for i in range(len(table[0]))]
    # Print the top border of the table
    print("+" + "+".join(['-' * (width + 2) for width in col_width]) + "+")
    # Print the header row
    for i, row in enumerate(table):
        print("|" + "|".join([' {:{}} '.format(str(row[j]), col_width[j]) for j in range(len(row))]) + "|")
        if i == 0:
            # Print the separator between the header and the data rows
            print("+" + "+".join(['=' * (width + 2) for width in col_width]) + "+")
        else:
            # Print the dotted line separator between the data rows
            print("+" + "+".join(['.' * (width + 2) for width in col_width]) + "+")
    # Print the bottom border of the table
    print("+" + "+".join(['-' * (width + 2) for width in col_width]) + "+")



BTC = yf.download('BTC-USD', start=datetime(2022, 4, 1), end=datetime(2023, 4, 1))
ETH = yf.download('ETH-USD',start=datetime(2022, 4, 1), end=datetime(2023, 4, 1))
# print(BTC.columns)
# print(ETH.columns)
pair= pd.concat([BTC['Adj Close'],ETH['Adj Close']], ignore_index=True,axis=1)
pair=pair.dropna()

# pair_feture=pd.concat([BTC['Open', 'High', 'Low', 'Close', 'Volume'],ETH['Open', 'High', 'Low', 'Close', 'Volume']], ignore_index=True,axis=1)
# pair_feature = pd.concat([BTC[['Open', 'High', 'Low', 'Close', 'Volume']], ETH[['Open', 'High', 'Low', 'Close', 'Volume']]], ignore_index=True, axis=1)
# pair_feature.cloumns=['BTC_Open_R', 'BTC_High_R', 'BTC_Low_R', 'BTC_Close_R', 'BTC_Volume_R','ETH_Open_R', 'ETH_High_R', 'ETH_Low_R', 'ETH_Close_R', 'ETH_Volume_R']

pair_feature = pd.concat([BTC[['Open', 'High', 'Low', 'Close', 'Volume']], ETH[['Open', 'High', 'Low', 'Close', 'Volume']]], ignore_index=True, axis=1)
pair_feature.columns=['BTC_Open_R', 'BTC_High_R', 'BTC_Low_R', 'BTC_Close_R', 'BTC_Volume_R', 'ETH_Open_R', 'ETH_High_R', 'ETH_Low_R', 'ETH_Close_R', 'ETH_Volume_R']

pair_feature_ratio=normalize_series(pair_feature)

pair_ret=normalize_series(pair)

#remove first row with NAs
pair_ret=pair_ret.tail(len(pair_ret)-1)
pair_ret.columns = ['BTC_RET','ETH_RET']

# split into train and validation/testing
# split=int(len(pair_ret) * 0.7)

btc_R_train = pair_ret['BTC_RET']
btc_R_test = pair_ret['BTC_RET']
eth_R_train = pair_ret['ETH_RET']
eth_R_test = pair_ret['ETH_RET']

# tests= pd.concat([btc_R_test ,eth_R_test,pair_feature_ratio], ignore_index=False,axis=1)


#z_score=(pair_spread-spread_mean)/spread_sd

hege= rg.OLS(btc_R_train, eth_R_train).fit().params[0]
# hege=1
pair_train= btc_R_test - hege * eth_R_test
# BTC_ETH Rolling Spread Z-Score Calculation


#getting equalent values of the returns, introducing return on                                                                                                    returns to build signals
rbtc_ret= pair_ret['BTC_RET']
reth_ret= pair_ret['ETH_RET']

tests_for_lable= pd.concat([btc_R_test ,eth_R_test,pair_feature_ratio], ignore_index=False,axis=1)
tests_for_lable = tests_for_lable.dropna()
a =  1.536870315595765
b = 0.07280825856722152
k = 2
window1 = 16
window2 = 80

z_score,_ = get_z_socre_two_windows(btc_R_train, eth_R_train,btc_R_test,eth_R_test,window1,window2)

z_score = z_score.dropna()

z_score_ret = triple_barrier(z_score, a, b, k)

z_score_singel_for_lable = z_score_ret['triple_barrier_signal']

# tests.insert(len(tests.columns), 'ftestsig2', z_score_singel)
tests_for_lable.insert(len(tests_for_lable.columns), 'rbtc_ret', rbtc_ret)
tests_for_lable.insert(len(tests_for_lable.columns), 'reth_ret', reth_ret)
tests_for_lable.insert(len(tests_for_lable.columns), 'z_score', z_score)
tests_for_lable.insert(len(tests_for_lable.columns), 'z_score(-1)', z_score.shift(1))
tests_for_lable.insert(len(tests_for_lable.columns), 'z_score(-2)', z_score.shift(2))
tests_for_lable.insert(len(tests_for_lable.columns), 'z_score_singel_for_lable', z_score_singel_for_lable)

# 3.3.6 Trading Strategy Signals, without commission/exchange fee

port_out_z_score_singel_for_lable = 0.0
port_outa_z_score_singel_for_lable = []

for i in range(0, len(tests_for_lable.index)):
    if tests_for_lable.at[tests_for_lable.index[i], 'z_score_singel_for_lable'] == 1:
        '''
        If the value of the z-score touches the upper threshold, 
        indicating a positive deviation from the mean, 
        it means that the growth rate of BTC is too fast. Therefore,
        it is recommended to buy ETH.
        '''
        port_out_z_score_singel_for_lable = tests_for_lable.at[tests_for_lable.index[i], 'reth_ret']
    elif tests_for_lable.at[tests_for_lable.index[i], 'z_score_singel_for_lable'] == -1:
        '''
        If the value of z_score touches the lower barrier, 
        indicating a negative deviation from the mean, 
        it means that the growth rate of ETH is too fast. 
        Therefore, buy BTC.
        '''
        port_out_z_score_singel_for_lable = tests_for_lable.at[tests_for_lable.index[i], 'rbtc_ret']
    else:
        port_out_z_score_singel_for_lable = 0
    port_outa_z_score_singel_for_lable.append(port_out_z_score_singel_for_lable)

# tests_for_lable.insert(len(tests_for_lable.columns), 'Log_R', np.log(1 + pd.DataFrame(port_outa_z_score_singel_for_lable)))
tests_for_lable.insert(len(tests_for_lable.columns), 'port_outa_z_score_singel_for_lable', port_outa_z_score_singel_for_lable)


tests_for_lable = tests_for_lable.fillna(method='ffill')

port_outa_z_score_singel_for_lable = (1 + tests_for_lable['port_outa_z_score_singel_for_lable']).cumprod() # pair trading return


# pt_out = port_outa_z_score_singel.iloc[3:]
MDD = get_mdd(port_outa_z_score_singel_for_lable)
print("------FINALLY---------------------------------------")
print("Return : " + str(np.round(port_outa_z_score_singel_for_lable.iloc[-1], 4)))
print("Standard Deviation : " + str(
    np.round(np.std(port_outa_z_score_singel_for_lable), 4)))  # mean_absolute_percentage_error
print("Sharpe Ratio (Rf=0%) : " + str(
    np.round(port_outa_z_score_singel_for_lable.iloc[-1] / (np.std(port_outa_z_score_singel_for_lable)), 4)))
print("Max Drawdown: " + str(np.round(MDD, 4)))  # calculate_mdd(pt_out)
print('++++++++++++++++++++++++++++++++++++++')
print("a : " + str(a))
print("b : " + str(b))
print("k : " + str(k))
print("window1 : " + str(window1))
print("window2 : " + str(window2))



pt_out_pair_trading=get_pairstrategy_return()

bh_btc= (1 + tests_for_lable['rbtc_ret']).cumprod()
bh_eth= (1 + tests_for_lable['reth_ret']).cumprod()



plt.figure(figsize=(16,8))
plt.rcParams.update({'font.size':10})
plt.xticks(rotation=45)
ax = plt.axes()
ax.xaxis.set_major_locator(plt.MaxNLocator(20))
plt.plot(pt_out_pair_trading, label='Cumulative return on P-Trading Strategy portfolio',color='b')
plt.plot(port_outa_z_score_singel_for_lable, label='Labeling Method',color='r')
# plt.plot(port_z_scorec, label='Cumulative return on Labeling_+5%Cm',color='y')
# plt.plot(pt_outc, label='Cumulative return on P-Trading Strategy_+5%Cm')
plt.plot(bh_btc, label='Cumulative return on Buy and Hold Bitcoin',color='g')
plt.plot(bh_eth, label='Cumulative return on Buy and Hold Ethereum',color='Purple')
plt.title('Labeling Method VS. P-Trading Strategy Cumulative Returns')
plt.xlabel("Date")
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.suptitle('Labeling Method VS. P-Trading Strategy Cumulative Returns')
ax.legend(loc='best')
ax.grid(True)
plt.show()



results2 = [{'0': 'Test:',
             '1': 'P-Trading Strategy',
             '2': 'triple barrier labeling',
             '3': 'Buy&Hold Bitcoin',
             '4': 'Buy&Hold Ethereum',},

            {'0': 'Return',
             '1': np.round(pt_out_pair_trading.iloc[-1], 4),
             '2': np.round(port_outa_z_score_singel_for_lable.iloc[-1], 4),
             '3': np.round(bh_btc.iloc[-1], 4),
             '4': np.round(bh_eth.iloc[-1], 4),
             # '6': np.round(pt_outc.iloc[-1], 4)
             },

            {'0': 'Standard Deviation',
             '1': np.round(np.std(pt_out_pair_trading), 4),
             '2': np.round(np.std(port_outa_z_score_singel_for_lable), 4),
             '3': np.round(np.std(bh_btc), 4),
             '4': np.round(np.std(bh_eth), 4),
             # '6': np.round(np.std(pt_outc), 4)
             },

            {'0': 'Sharpe Ratio (Rf=0%)',
             '1': np.round(pt_out_pair_trading.iloc[-1] / (np.std(pt_out_pair_trading)), 4),
             '2': np.round(port_outa_z_score_singel_for_lable.iloc[-1] / (np.std(port_outa_z_score_singel_for_lable)), 4),
             '3': np.round(bh_btc.iloc[-1] / (np.std(bh_btc)), 4),
             '4': np.round(bh_eth.iloc[-1] / (np.std(bh_eth)), 4),
             # '6': np.round(pt_outc.iloc[-1] / (np.std(pt_outc)), 4)
             },

            {'0': 'Max Drawdown',
             '1': np.round(get_mdd(pt_out_pair_trading), 4),
             '2': np.round(get_mdd(port_outa_z_score_singel_for_lable), 4),
             '3': np.round(get_mdd(bh_btc), 4),
             '4': np.round(get_mdd(bh_eth), 4),
             # '6': np.round(get_my_mdd(pt_outc), 4)
             }
            ]





table2 = pd.DataFrame(results2)
print_table(table2.values.tolist())

tests_for_lable.insert(len(tests_for_lable.columns), 'Log_R', np.log(1 + pd.DataFrame(port_outa_z_score_singel_for_lable)))
tests_for_lable=tests_for_lable.dropna()
tests_for_lable.to_csv("tests_for_lable_dataset.csv", index=True)

# #-------------------------------------------------pair trading-----------------------------------------
# BTC = yf.download('BTC-USD', start=datetime(2017, 11, 9), end=datetime(2023, 5, 14))
# ETH = yf.download('ETH-USD', start=datetime(2017, 11, 9), end=datetime(2023, 5, 14))
# pair = pd.concat([BTC['Adj Close'], ETH['Adj Close']], ignore_index=True, axis=1)
# pair = pair.dropna()
# pair_ret = normalize_series(pair)
# pair_ret = pair_ret.tail(len(pair_ret) - 1)
# pair_ret.columns = ['BTC_RET', 'ETH_RET']
#
# split=326 #int(len(pair_ret) * 0.7)
#
# btc_R_train = pair_ret['BTC_RET'][:split]
# btc_R_test = pair_ret['BTC_RET'][split:]
# eth_R_train = pair_ret['ETH_RET'][:split]
# eth_R_test = pair_ret['ETH_RET'][split:]
# tests= pd.concat([btc_R_test ,eth_R_test], ignore_index=False,axis=1)
#
# pair_spread= btc_R_test - rg.OLS(btc_R_train, eth_R_train).fit().params[0] * eth_R_test
# #calculating for beta (Hedge)
# beta= rg.OLS(btc_R_train, eth_R_train).fit().params[0]
#
# #Spread Mean and Standard dev
# spread_mean= pair_spread.mean()
# spread_sd= pair_spread.std()
# print('the mean of the spread is', spread_mean)
# print('the Standard Dev of the Spread is',spread_sd)
# #z_score=(pair_spread-spread_mean)/spread_sd
# window= 42 #29 54
# pair_train= btc_R_test - rg.OLS(btc_R_train, eth_R_train).fit().params[0] * eth_R_test
# # BTC_ETH Rolling Spread Z-Score Calculation
# z_score = (pair_train - pair_train.rolling(window=window,min_periods=1).mean()) / pair_train.rolling(window=window,min_periods=1).std()
# # z_score,_ = get_z_socre_two_windows(btc_R_train, eth_R_train,btc_R_test,eth_R_test,window1,window2)
#
# z_score=z_score.dropna()
# # print(z_score.rolling(window=2,min_periods=1).mean())
# z_score_mean=z_score.rolling(window=2,min_periods=1).mean()
# # z_score_mean=z_score_mean[1:]
# # print(z_score.rolling(window=2,min_periods=1).std())
# z_score_std=z_score.rolling(window=2,min_periods=1).std()
# z_score_std.dropna()
#
# up_th = (z_score.rolling(window=2).mean())+(z_score.rolling(window=2).std()*2) # upper threshold
# lw_th = (z_score.rolling(window=2).mean())-(z_score.rolling(window=2).std()*2) # lower threshold
#
# up_th=up_th.dropna()
# lw_th=lw_th.dropna()
# up_lw= pd.concat([up_th,lw_th, btc_R_test ,eth_R_test,z_score,z_score.rolling(window=2).mean(),z_score.rolling(window=2).std()*2], ignore_index=True,axis=1)
# up_lw.columns = [ 'up_th','lw_th','btc_R_test','eth_R_test','z_score','mean(window=2)','deviation(window=2)']
# up_lw.dropna()
# rbtc_ret= pair_ret['BTC_RET'].tail(1693)  # picking the last 120 transactions on Bitcoin  for our trading
# reth_ret= pair_ret['ETH_RET'].tail(1693)  # picking the last 120 transactions on Ethereum  for our trading
# #return on returns
# rrbtc=(pair_ret['BTC_RET'].pct_change(1).dropna()).pct_change(1).dropna()#return on returns for bitcoin
#
# rreth=(pair_ret['ETH_RET'].pct_change(1).dropna()).pct_change(1).dropna()#return on returns for ethereum
#
# trade_dir= pd.DataFrame(rbtc_ret)
# trade_dir.insert(len(trade_dir.columns), 'rbtc_ret(-1)', rbtc_ret.shift(1))
# trade_dir.insert(len(trade_dir.columns), 'rbtc_ret(-2)', rbtc_ret.shift(2))
# trade_dir.insert(len(trade_dir.columns), 'reth_ret(-1)', reth_ret.shift(1))
# trade_dir.insert(len(trade_dir.columns), 'reth_ret(-2)', reth_ret.shift(2))
# trade_dir.insert(len(trade_dir.columns), 'rrbtc(-1)', rrbtc.shift(1))
# trade_dir.insert(len(trade_dir.columns), 'rrbtc(-2)', rrbtc.shift(2))
# trade_dir.insert(len(trade_dir.columns), 'rreth(-1)', rreth.shift(1))
# trade_dir.insert(len(trade_dir.columns), 'rreth(-2)', rreth.shift(2))
# trade_dirsig2 = 0.0
# trade_dirsig2a = []
#
# trade_dir=trade_dir.dropna()
# for i in trade_dir.index.values:
#     if trade_dir.at[trade_dir.index[i], 'rrbtc(-2)'] > (rreth[trade_dir.index[i]]) and trade_dir.at[
#         trade_dir.index[i], 'rrbtc(-1)'] < (rreth[trade_dir.index[i]]):
#         trade_dirsig2 = 2
#     elif trade_dir.at[trade_dir.index[i], 'rrbtc(-2)'] < (rreth[trade_dir.index[i]]) and trade_dir.at[
#         trade_dir.index[i], 'rrbtc(-1)'] > (rreth[trade_dir.index[i]]):
#         trade_dirsig2 = -2
#     elif trade_dir.at[trade_dir.index[i], 'rreth(-2)'] > (rrbtc[trade_dir.index[i]]) and trade_dir.at[
#         trade_dir.index[i], 'rreth(-1)'] < (rrbtc[trade_dir.index[i]]):
#         trade_dirsig2 = 1
#     elif trade_dir.at[trade_dir.index[i], 'rreth(-2)'] < (rrbtc[trade_dir.index[i]]) and trade_dir.at[
#         trade_dir.index[i], 'rreth(-1)'] > (rrbtc[trade_dir.index[i]]):
#         trade_dirsig2 = -1
#     else:
#         trade_dirsig2 = 0.0
#     trade_dirsig2a.append(trade_dirsig2)
#
# trade_dir.insert(len(trade_dir.columns), 'trade_dirsig2', trade_dirsig2a)
#
#
# #3.3.5. BTC_ETH Trading Strategy Signals
# tests.insert(len(tests.columns), 'z_score', z_score)
# tests.insert(len(tests.columns), 'z_score(-1)', z_score.shift(1))
# tests.insert(len(tests.columns), 'z_score(-2)', z_score.shift(2))
# tests.insert(len(tests.columns), 'trade_dir', trade_dir['trade_dirsig2'])
#
# tests=tests.dropna()
#
# ftestsig2 = 0.0
# ftestsig2a = []
# for i in tests.index.values:
#     if tests.at[tests.index[i], 'z_score(-2)'] > (-1 * up_th[tests.index[i]]) and tests.at[
#         tests.index[i], 'z_score(-1)'] < (-1 * up_th[tests.index[i]]):
#         ftestsig2 = 1
#     elif tests.at[tests.index[i], 'z_score(-2)'] < (-1 * lw_th[tests.index[i]]) and tests.at[
#         tests.index[i], 'z_score(-1)'] > (-1 * lw_th[tests.index[i]]):
#         ftestsig2 = -2
#     elif tests.at[tests.index[i], 'z_score(-2)'] < (-1 * up_th[tests.index[i]]) and tests.at[
#         tests.index[i], 'z_score(-1)'] > (-1 * up_th[tests.index[i]]):
#         ftestsig2 = -1
#     elif tests.at[tests.index[i], 'z_score(-2)'] > (-1 * lw_th[tests.index[i]]) and tests.at[
#         tests.index[i], 'z_score(-1)'] < (-1 * lw_th[tests.index[i]]):
#         ftestsig2 = 2
#     elif tests.at[tests.index[i], 'z_score(-2)'] < up_th[tests.index[i]] and tests.at[
#         tests.index[i], 'z_score(-1)'] > up_th[tests.index[i]]:
#         ftestsig2 = -1
#     elif tests.at[tests.index[i], 'z_score(-2)'] > up_th[tests.index[i]] and tests.at[
#         tests.index[i], 'z_score(-1)'] < up_th[tests.index[i]]:
#         ftestsig2 = 1
#     elif tests.at[tests.index[i], 'z_score(-2)'] > lw_th[tests.index[i]] and tests.at[
#         tests.index[i], 'z_score(-1)'] < lw_th[tests.index[i]]:
#         ftestsig2 = 2
#     elif tests.at[tests.index[i], 'z_score(-2)'] < lw_th[tests.index[i]] and tests.at[
#         tests.index[i], 'z_score(-1)'] > lw_th[tests.index[i]]:
#         ftestsig2 = -2
#     elif tests.at[tests.index[i], 'trade_dir'] == 1:
#         ftestsig2 = 2
#     elif tests.at[tests.index[i], 'trade_dir'] == -1:
#         ftestsig2 = -2
#     elif tests.at[tests.index[i], 'trade_dir'] == 2:
#         ftestsig2 = 1
#     elif tests.at[tests.index[i], 'trade_dir'] == -2:
#         ftestsig2 = -1
#     else:
#         ftestsig2 = 0.0
#     ftestsig2a.append(ftestsig2)
#
# tests.insert(len(tests.columns), 'rbtc_ret', rbtc_ret)
# tests.insert(len(tests.columns), 'reth_ret', reth_ret)
#
# port_out = 0.0
# port_outa = []
# tests.insert(len(tests.columns), 'ftestsig2', ftestsig2a)
# for i in tests.index.values:
#     if tests.at[tests.index[i], 'ftestsig2'] == -2:
#         port_out = tests.at[tests.index[i], 'rbtc_ret']
#     elif tests.at[tests.index[i], 'ftestsig2'] == -1:
#         port_out = tests.at[tests.index[i], 'reth_ret']
#     elif tests.at[tests.index[i], 'ftestsig2'] == 2:
#         port_out = tests.at[tests.index[i], 'rbtc_ret']
#     elif tests.at[tests.index[i], 'ftestsig2'] == 1:
#         port_out = tests.at[tests.index[i], 'reth_ret']
#     else:
#         port_out = 0
#     port_outa.append(port_out)
#     port_outa.append(port_out)
# tests.insert(len(tests.columns), 'port_out', port_outa)
# tests = tests.fillna(method='ffill')
#
# pt_out_pair_trading = (1 + tests['port_out']).cumprod()
#
# bh_btc= (1 + tests['rbtc_ret']).cumprod()         #np.exp(np.log1p(tests['rbtc_ret']).cumsum())- 1 # buy and Hold Bitcoin
# bh_eth= (1 + tests['reth_ret']).cumprod()         #np.exp(np.log1p(tests['reth_ret']).cumsum())- 1 # buy and Hold Ethereum
#
#
# merged_two = tests.merge(tests_for_lable, how='inner', left_index=True, right_index=True)
#
# merged_df = merged_two.loc[:, ~merged_two.columns.duplicated()]
#
# pt_out_pair_trading = (1 + merged_df['port_out']).cumprod()
#
# temp_pair=np.round(get_mdd(pt_out_pair_trading), 4)
#
# pt_out_for_label = (1 + merged_df['port_outa_z_score_singel_for_lable']).cumprod()
#
# temp_for_label=np.round(get_mdd(pt_out_for_label), 4)
#
#
# #-------------------------------------------------pair trading-----------------------------------------
#
# plt.figure(figsize=(16,8))
# plt.rcParams.update({'font.size':10})
# plt.xticks(rotation=45)
# ax = plt.axes()
# ax.xaxis.set_major_locator(plt.MaxNLocator(20))
# plt.plot(pt_out_pair_trading, label='Cumulative return on P-Trading Strategy portfolio',color='b')
# plt.plot(pt_out_for_label, label='Labeling Method',color='r')
# # plt.plot(port_z_scorec, label='Cumulative return on Labeling_+5%Cm',color='y')
# # plt.plot(pt_outc, label='Cumulative return on P-Trading Strategy_+5%Cm')
# plt.plot(bh_btc, label='Cumulative return on Buy and Hold Bitcoin',color='g')
# plt.plot(bh_eth, label='Cumulative return on Buy and Hold Ethereum',color='Purple')
# plt.title('BTC_ETH Trading Strategy Cumulative Returns')
# plt.xlabel("Date")
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.suptitle('BTC_ETH Portfolio Cumulative Returns  (120 days ( 55-days rolling window))')
# ax.legend(loc='best')
# ax.grid(True)
# plt.show()
#
#
# results2 = [{'0': 'Test:',
#              '1': 'P-Trading Strategy',
#              '2': 'triple barrier labeling',
#              '3': 'Buy&Hold Bitcoin',
#              '4': 'Buy&Hold Ethereum',},
#
#             {'0': 'Return',
#              '1': np.round(pt_out_pair_trading.iloc[-1], 4),
#              '2': np.round(pt_out_for_label.iloc[-1], 4),
#              '3': np.round(bh_btc.iloc[-1], 4),
#              '4': np.round(bh_eth.iloc[-1], 4),
#              # '6': np.round(pt_outc.iloc[-1], 4)
#              },
#
#             {'0': 'Standard Deviation',
#              '1': np.round(np.std(pt_out_pair_trading), 4),
#              '2': np.round(np.std(pt_out_for_label), 4),
#              '3': np.round(np.std(bh_btc), 4),
#              '4': np.round(np.std(bh_eth), 4),
#              # '6': np.round(np.std(pt_outc), 4)
#              },
#
#             {'0': 'Sharpe Ratio (Rf=0%)',
#              '1': np.round(pt_out_pair_trading.iloc[-1] / (np.std(pt_out_pair_trading)), 4),
#              '2': np.round(pt_out_for_label.iloc[-1] / (np.std(pt_out_for_label)), 4),
#              '3': np.round(bh_btc.iloc[-1] / (np.std(bh_btc)), 4),
#              '4': np.round(bh_eth.iloc[-1] / (np.std(bh_eth)), 4),
#              # '6': np.round(pt_outc.iloc[-1] / (np.std(pt_outc)), 4)
#              },
#
#             {'0': 'Max Drawdown',
#              '1': np.round(get_mdd(pt_out_pair_trading), 4),
#              '2': np.round(get_mdd(pt_out_for_label), 4),
#              '3': np.round(get_mdd(bh_btc), 4),
#              '4': np.round(get_mdd(bh_eth), 4),
#              # '6': np.round(get_my_mdd(pt_outc), 4)
#              }
#             ]
#
#
#
#
#
# table2 = pd.DataFrame(results2)
# print_table(table2.values.tolist())

