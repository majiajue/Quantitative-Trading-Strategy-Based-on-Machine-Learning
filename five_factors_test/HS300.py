# -*- coding: utf-8 -*-
from atrader import *
from atrader.enums import * 
import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.datasets import load_iris, load_digits, load_boston
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import accuracy_score
import pickle
from pandas.core.frame import DataFrame
import talib

def init(context: Context):  
    set_backtest(initial_cash=1e7, stock_cost_fee=30)

    context.tarlist=get_code_list('hs300')
    reg_kdata(frequency='day', fre_num=1, adjust=False)
    context.ratio = 1   #初始权重设为1
    begin_date = '2016-01-01'
    end_date = '2016-03-31'

    context.long = 60
    context.short = 7
    context.Len = 10  # 用于计算波动（标准差），长度为context.Len = 10

    context.days = len(get_trading_days(market='sse', begin_date=begin_date, end_date=end_date))
    days = get_trading_days(market='sse', begin_date=begin_date, end_date=end_date)
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()

def on_data(context: Context):
    if datetime.datetime.strftime(context.now, '%Y-%m-%d') not in context.month_begin:
        return
    price = get_reg_kdata(reg_idx=context.reg_kdata[0], length=1, fill_up=True, df=True)
    index = get_reg_kdata(reg_idx=context.reg_kdata[0],target_indices=300, length=context.long+context.Len-1, fill_up=False, df=True)
    print(index)


    index['ret'] = index.groupby('target_idx')['close'].apply(lambda x: (x - x.shift()) / x.shift())
    index = index.fillna(0)  # 将NaN换为0
    ret = index.ret.values.astype(float)
    StdDev = talib.STDDEV(ret, timeperiod=context.Len, nbdev=1)
    StdDev = DataFrame({"a": StdDev})
    StdDev = StdDev.dropna()
    std = StdDev['a'].tolist()
    std_short = np.mean(std[-14:])
    bound = np.mean(std)
    print(std_short)

if __name__ == "__main__":
    # 投资域
    begin_date = '2015-01-01'
    end_date = '2016-03-31'
    cal=get_trading_days(market='sse', begin_date=begin_date, end_date=end_date)
    warnings.filterwarnings("ignore")
    tarlist = get_code_list('hs300')
    target_list = tarlist['code'].tolist() + ['sse.000300']
    a = run_backtest('HS300',target_list=target_list,file_path='.', begin_date=begin_date,
               end_date=end_date,frequency='day', fq=1)
