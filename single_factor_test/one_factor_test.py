# *_*coding:utf-8 *_*  
from atrader import *  
import numpy as np
import pandas as pd
import time
import datetime
import warnings

starttime = datetime.datetime.now()

def init(context):  

    set_backtest(initial_cash=1e7, stock_cost_fee=30)
    context.factor_info = {'PB': False}
    reg_factor(factor=list(context.factor_info.keys()))
    days = get_trading_days('SSE', '2016-01-01', '2018-9-30')
    months = np.vectorize(lambda x: x.month)(days)
    month_begin = days[pd.Series(months) != pd.Series(months).shift(1)]
    context.month_begin = pd.Series(month_begin).dt.strftime('%Y-%m-%d').tolist()

# 对因子进行排序，获取目标资产
def get_target_sec(context):
    data = get_reg_factor(reg_idx=context.reg_factor[0],length=1, df=True)

    data.dropna(inplace=True)
    # 注意将字典值对象转换为列表取出list(context.factor_info.values())[0]
    print(context.now)

    # 预处理，去极值
    mean = data.mean()['value']   # data的均值
    sigma = data.std()['value']   # data的标准差
    upper_bound = mean + 3 * sigma  # 上界
    lower_bound = mean - 3 * sigma  # 下界
    total_idx = data.shape[0]
    for i in range(total_idx):
        if data.iloc[i, 3] > upper_bound:
            data.iloc[i, 3] = upper_bound
        elif data.iloc[i, 3] < lower_bound:
            data.iloc[i, 3] = lower_bound
        else:
            pass

    # 预处理，标准化
    data['value'] = (data['value'] - mean)/sigma

    data.sort_values(by='value', ascending=list(context.factor_info.values())[0], inplace=True)
    print(list(data['target_idx'][:10]))
    #print(list(data['value'][:10]))

    return list(data['target_idx'][:10])
    
def order(context, target_sec):

    # 平掉所有持仓
    order_close_all(account_idx=0)
    # 买入
    percent = context.ratio = 1/len(target_sec)
    for target in target_sec :
        order_percent(account_idx=0, target_idx=target, percent=percent,
                      position_effect=1, side=1, order_type=2)


def on_data(context):
    if datetime.datetime.strftime(context.now, '%Y-%m-%d') not in context.month_begin:  # 调仓频率为月
        return
    # 获取将要购买的股票id
    target_sec = get_target_sec(context)
    warnings.filterwarnings("ignore")
    # 进行买卖操作
    order(context, target_sec) 


if __name__ == '__main__':  
    # 获取上证50的成分股  
    begin = '2016-01-01'
    end ='2018-09-30'
    cons_date = datetime.datetime.strptime(begin, '%Y-%m-%d') - datetime.timedelta(days=1)
    target = get_code_list('hs300',cons_date)
    target = list(target['code'])
    id = run_backtest(strategy_name='PB', file_path='.',
                      target_list=target, frequency='day', fre_num=1, 
                      begin_date=begin, end_date=end)
    warnings.filterwarnings("ignore")
    endtime = datetime.datetime.now()
    print('单因子测试总时长：'+str(endtime - starttime))
