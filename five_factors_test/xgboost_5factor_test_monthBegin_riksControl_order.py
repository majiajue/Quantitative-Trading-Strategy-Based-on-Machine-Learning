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

"""
2016-01-01至2018-03-31训练，2018-03-31至2019-03-31测试,全分开，不滚动
"""
def init(context: Context):
    set_backtest(initial_cash=1e7, stock_cost_fee=30)
    context.faclist = ['LFLO',
                       'PB',
                       'NIAP',
                       'NegMktValue',
                       'MktValue'
                       ]
    context.tarlist=get_code_list('hs300')
    reg_factor(context.faclist)
    reg_kdata(frequency='day', fre_num=1, adjust=False)

    context.ratio = 1.0   #初始权重设为1

    begin_date = '2016-01-01'
    end_date = '2019-03-31'
    context.cal = pd.Series(get_trading_days(market='sse', begin_date=begin_date, end_date=end_date))
    # get_trading_days()以列表的形式获取2016-03-31到2018-03-31的交易日期（包含2018-03-31）

    days = get_trading_days(market='sse', begin_date=begin_date, end_date=end_date)
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()

    context.long = 60
    context.short = 7
    context.Len = 10  # 用于计算波动（标准差），长度为context.Len = 10

def on_data(context: Context):

    if datetime.datetime.strftime(context.now, '%Y-%m-%d') not in context.month_begin:
        return
    # 获取沪深300指数数据
    price = get_reg_kdata(reg_idx=context.reg_kdata[0], length=1, fill_up=True, df=True)
    index = get_reg_kdata(reg_idx=context.reg_kdata[0],target_indices=300, length=context.long+context.Len-1, fill_up=False, df=True)
    factor = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=(), length=5, df=True)
    if price['close'].isna().any():
        return

    """
    计算沪深300指数的长短期波动率，以长期波动率为门限，若短期波动率突破，
    则降低股票池持仓为50%
    """
    index['ret'] = index.groupby('target_idx')['close'].apply(lambda x: (x - x.shift()) / x.shift())
    index = index.fillna(0)  # 将NaN换为0
    ret = index.ret.values.astype(float)
    StdDev = talib.STDDEV(ret, timeperiod=context.Len, nbdev=1)
    StdDev = DataFrame({"a": StdDev})
    StdDev = StdDev.dropna()
    std = StdDev['a'].tolist()
    std_short = np.mean(std[-14:])
    bound = np.mean(std)

    # factor的注册频率默认为日频
    factor = factor.dropna(subset =['date'])   # 删除非法日期
    factor['code'] = factor['target_idx'].apply(lambda x:context.target_list[x])    # 将用0，1，2，3等表示的股票换成对应的股票代码
    factor['month'] = factor['date'].apply(lambda x:int(str(x)[0:4]+str(x)[5:7]))  # 增加month列，2017-01，2017-02，只记录月份，不记录日时分秒
    factor_name = factor['factor'].drop_duplicates().tolist()   # 以列表的形式取出因子名称
    # 将factor按['target_idx','month','factor']分组，分别取每组的最后一行
    # 即取出各股票每个月末的所有因子值
    factor_month = factor.groupby(['target_idx','month','factor']).apply(lambda x:x.iloc[-1])[['date','value']].reset_index()
    # 添加所有因子名作为新的列
    factor_month1 = factor_month.groupby(['target_idx','month']).apply(deal).reset_index()

    """
    取最后一个月(当前时间)
    """
    test = factor_month1.groupby('target_idx').apply(lambda x: x.iloc[-1])
    scaler = StandardScaler()   # 标准化

    X_test = test[factor_name]
    X_test = X_test.fillna(0).values
    #X_test=scaler.fit_transform(X_test)      # 因子标准化
    X_test = scaler.fit_transform(X_test)  # 因子标准化

    # 预测
    model = pickle.load(open("XGboost_ret0.06_5factor.pickle.dat", "rb"))
    y_pred = model.predict(X_test)
    y_pred1 = pd.DataFrame(y_pred,columns=['label'])
    idx_list = list(y_pred1[y_pred1['label']==1].index)
    print(idx_list)

    positions = context.account().positions
    if len(idx_list) == 0:  # 没有一只股票在标的池，则卖出全部股票
        for target_idx in positions.loc[positions['volume_long'] > 0, 'target_idx'].astype(int):
            volume = positions['volume_long'].iloc[target_idx]
            order_volume(account_idx=0, target_idx=target_idx,
                         volume=int(volume),
                         side=2, position_effect=2, order_type=2, price=0)

    else:
        positions = context.account().positions
        # 平不在标的池的股票
        for target_idx in positions.target_idx.astype(int):
            if target_idx not in idx_list:
                if positions['volume_long'].iloc[target_idx] > 0:
                    volume = positions['volume_long'].iloc[target_idx]
                    order_volume(account_idx=0, target_idx=target_idx,
                                 volume=int(volume),side=2, position_effect=2, order_type=2, price=0)
                    print("平不在标的池的股票"+str(target_idx))

        # 根据波动率进行风险控制
        if std_short > bound:
            positions = context.account().positions
            for target_idx in positions.loc[positions['volume_long'] > 0, 'target_idx'].astype(int):
                volume = positions['volume_long'].iloc[target_idx]
                order_volume(account_idx=0, target_idx=target_idx,
                             volume=int(volume * 0.5),
                             side=2, position_effect=2, order_type=2, price=0)
                print("风险控制" + str(target_idx))
            # 获取股票的权重
            positions = context.account().positions
            percent_b = context.ratio / len(idx_list)
            # print(percent_b)
            # 买在标的池中的股票
            for target_idx in idx_list:
                if target_idx == 300:
                    pass
                else:
                    order_target_percent(account_idx=0, target_idx=target_idx, target_percent=percent_b*0.5, side=1, order_type=2)
                print(positions.loc[positions['volume_long'] > 0, 'code'].tolist())
        else:
            # 获取股票的权重
            positions = context.account().positions
            percent_b = context.ratio / len(idx_list)
            # print(percent_b)
            # 买在标的池中的股票
            for target_idx in idx_list:
                if target_idx == 300:
                    pass
                else:
                    order_target_percent(account_idx=0, target_idx=target_idx, target_percent=percent_b*0.5, side=1, order_type=2)
            print(positions.loc[positions['volume_long'] > 0, 'code'].tolist())


def deal(df):
    factor_name = df['factor'].tolist()
    df1 = pd.DataFrame(columns = factor_name)
    for i in factor_name:
        df1[i] = df.loc[df['factor']==i,'value'].values
    return df1

if __name__ == "__main__":
    # 投资域
    begin_date = '2016-01-01'
    end_date = '2019-03-31'
    cal=get_trading_days(market='sse', begin_date=begin_date, end_date=end_date)
    warnings.filterwarnings("ignore")
    tarlist = get_code_list('hs300')
    target_list = tarlist['code'].tolist() + ['sse.000300']

    a = run_backtest('XGBoost_2016_2019_风控',target_list=target_list,file_path='.', begin_date=begin_date,
               end_date=end_date,frequency='day', fq=1)
