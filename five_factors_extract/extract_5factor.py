from atrader import *
from atrader.enums import *
import numpy as np
import pandas as pd
import datetime
import xgboost as xgb
import warnings

def init(context: Context):
    set_backtest(initial_cash=1e6)
    context.faclist = ['LFLO',
                       'KAMA',
                       'TotalAssetGrowRate',
                       'PB',
                       'NIAP'
                       ]
    context.tarlist=get_code_list('hs300')
    reg_factor(context.faclist)
    reg_kdata(frequency='month', fre_num=1, adjust=False)
    context.ratio = 0.9    #初始权重设为0.9
    begin_date = '2016-01-31'
    end_date = '2019-03-31'
    context.cal=pd.Series(get_trading_days(market='sse', begin_date=begin_date, end_date=end_date))
    # get_trading_days()以列表的形式获取2016-03-31到2018-03-31的交易日期（包含2018-03-31）
    # len取其长度，作为后面获取factor的天数
    context.days = len(get_trading_days(market='sse', begin_date=begin_date, end_date=end_date))

def on_data(context: Context):
    # price
    price = get_reg_kdata(reg_idx=context.reg_kdata[0], target_indices=(), length=33, fill_up=True, df=True)  #获取33个月的price

    # factor
    factor = get_reg_factor(reg_idx=context.reg_factor[0], target_indices=(), length=context.days, df=True)
    factor.to_csv("5factor.csv")

    price.to_csv("price_5factor_test.csv")

if __name__ == "__main__":

    # 投资域
    begin_date = '2016-01-31'
    end_date = '2019-03-31'
    cal=get_trading_days(market='sse', begin_date=begin_date, end_date=end_date)
    warnings.filterwarnings("ignore")
    start = datetime.datetime.now()
    tarlist = get_code_list('hs300')

    a = run_backtest('XGBoost',target_list=tarlist['code'].tolist(),file_path='.', begin_date=begin_date,
               end_date=end_date,frequency='month', fq=1)
