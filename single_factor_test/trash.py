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

def init(context: Context):  
    set_backtest(initial_cash=1e7, stock_cost_fee=30)
    context.faclist = ['KAMA']
    context.tarlist=get_code_list('hs300')
    reg_factor(context.faclist)
    reg_kdata(frequency='day', fre_num=1, adjust=False)
    context.ratio = 0.9    #初始权重设为0.9
    begin_date = '2016-01-01'
    end_date = '2018-09-30'
    context.cal = pd.Series(get_trading_days(market='sse', begin_date=begin_date, end_date=end_date))

def on_data(context: Context):
    print(context.now)
    # 实时模式返回当前本地时间, 回测模式返回当前回测时间
    now = context.now
    # begin_date1(每月1号), end_date1 相差一个月
    begin_date1=now.replace(day=1)
    if now.month==12:
        if now.year == 2016:
            end_date1=begin_date1.replace(year=2017).replace(month=1)
        if now.year == 2017:
            end_date1 = begin_date1.replace(year=2018).replace(month=1)
        if now.year == 2018:
            end_date1 = begin_date1.replace(year=2019).replace(month=1)
    else:
        end_date1=begin_date1.replace(month=(begin_date1.month+1))

    # call提取begin_date1与end_date1之间的时间
    cal1=context.cal[context.cal<end_date1]
    cal1=cal1[cal1>=begin_date1]

    if now<cal1.iloc[-2]:
        pass

    # 下月初
    else:
        # factor的注册频率默认为日频
        factor = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=(), length=1, df=True)  # 注册月末的因子值
        factor = factor.dropna(subset=['date'])  # 删除非法日期
        factor['code'] = factor['target_idx'].apply(lambda x: context.target_list[x])  # 添加对应的股票代码
        # 预处理，去极值
        mean = factor.mean()['value']  # data的均值
        sigma = factor.std()['value']  # data的标准差
        upper_bound = mean + 3 * sigma  # 上界
        lower_bound = mean - 3 * sigma  # 下界
        total_idx = factor.shape[0]
        for i in range(total_idx):
            if factor.iloc[i, 3] > upper_bound:
                factor.iloc[i, 3] = upper_bound
            elif factor.iloc[i, 3] < lower_bound:
                factor.iloc[i, 3] = lower_bound
            else:
                pass

        # 预处理，标准化
        factor['value'] = (factor['value'] - mean) / sigma

        factor.sort_values(by="value", ascending=False)
        context.idx_list = list(factor['target_idx'][0:10])
        print(context.idx_list)
        positions = context.account().positions

        # 平不在标的池的股票
        for target_idx in positions.target_idx.astype(int):
            if target_idx not in context.idx_list:
                order_volume(account_idx=0, target_idx=target_idx,
                             volume=int(positions['volume_long'].iloc[target_idx]),
                             side=2, position_effect=2, order_type=2, price=0)
                warnings.filterwarnings("ignore")

        # 买入再标的池的股票
        percent_b = context.ratio / len(context.idx_list)        # 获取股票的权重
        # 买在标的池中的股票
        for target_idx in context.idx_list:
            order_target_percent(account_idx=0, target_idx=target_idx, target_percent=percent_b, side=1, order_type=2)
            warnings.filterwarnings("ignore")
        print(positions.loc[positions['volume_long'] > 0, 'code'].tolist())

if __name__ == "__main__":
    # 投资域
    begin_date = '2016-01-01'
    end_date = '2018-09-30'
    cal=get_trading_days(market='sse', begin_date=begin_date, end_date=end_date)
    warnings.filterwarnings("ignore")
    start = datetime.datetime.now()
    tarlist = get_code_list('hs300')

    a = run_backtest('KAMA',target_list=tarlist['code'].tolist(),file_path='.', begin_date=begin_date,
               end_date=end_date,frequency='day', fq=1)
