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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import warnings


def init(context: Context):
    set_backtest(initial_cash=1e6)

    context.tarlist = get_code_list('hs300')

    reg_kdata(frequency='week', fre_num=1, adjust=False)
    context.ratio = 0.9
    begin_date = '2016-01-01'
    end_date = '2018-09-30'
    context.cal = pd.Series(get_trading_days(market='sse', begin_date=begin_date, end_date=end_date))
    # 设置开仓的最大资金量

def on_data(context: Context):
    # time_list = list(get_trading_days(market='sse', begin_date='2016-01-01', end_date='2017-06-30'))
    print(context.now)
    now = context.now

    begin_date1 = now.replace(day=1)
    if now.month == 12:
        end_date1 = begin_date1.replace(year=2018).replace(month=1)
    else:
        end_date1 = begin_date1.replace(month=((begin_date1.month + 1)))
    cal1 = context.cal[context.cal < end_date1]
    cal1 = cal1[cal1 > begin_date1]
    orderlist = get_order_info(order_list=())
    print(cal1)
    # cal[-2:-1]为本月最后一个交易日
    if now < datetime.datetime(2017, 2, 1):
        pass
    else:
        if now < cal1.iloc[-2]:
            ## 查询当前价格进行止损
            # price0=get_reg_kdata(reg_idx=context.reg_kdata[0], target_indices=(), length=1, fill_up=True, df=True)
            positions = context.account().positions
            idx_long_list = positions.loc[positions['volume_long'] > 0, 'target_idx']
            for idx in idx_long_list:
                # price_idx = price0.loc[price0['target_idx']==idx,'close']
                amount_long = positions['amount_long'].iloc[idx]
                fpnl_long = positions['fpnl_long'].iloc[idx]
                if -fpnl_long > amount_long * 0.1:
                    ##卖出
                    order_volume(account_idx=0, target_idx=idx,
                                 volume=int(positions['volume_long'].iloc[idx]),
                                 side=2, position_effect=2, order_type=2, price=0)
                else:
                    pass
        else:

            ## price
            price = get_reg_kdata(reg_idx=context.reg_kdata[0], target_indices=(), length=13, fill_up=True, df=True)
            print(price)
            price = price[price['close'] != 0]
            price['ret_month'] = price.groupby('target_idx')['close'].apply(lambda x: (x - x.shift()) / x.shift())
            price.loc[price['ret_month'] >= 0.03, 'label'] = 1
            price.loc[price['ret_month'] < 0.03, 'label'] = 0
            price_month1 = price[['target_idx', 'time', 'ret_month', 'close', 'label']]
            price_month1['month'] = price_month1['time'].apply(lambda x: int(str(x)[0:4] + str(x)[5:7]))
            price_month1['ret_nextmonth'] = price_month1.groupby('target_idx')['ret_month'].shift(-1)
            price_month1['label'] = price_month1.groupby('target_idx')['label'].shift(-1)
            print(price_month1)


if __name__ == "__main__":
    # 投资域
    begin_date = '2016-01-01'
    end_date = '2018-09-30'
    cal = get_trading_days(market='sse', begin_date=begin_date, end_date=end_date)
    warnings.filterwarnings("ignore")
    start = datetime.datetime.now()
    tarlist = get_code_list('hs300')


    a = run_backtest('EXP', target_list=tarlist['code'].tolist(), file_path='.',
                     begin_date=begin_date,
                     end_date=end_date, frequency='week', fq=1)


