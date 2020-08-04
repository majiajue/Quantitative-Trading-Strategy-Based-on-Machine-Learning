# *_*coding:utf-8 *_*
from atrader import *
from atrader.enums import *

def init(context: Context):
    # 注册三个因子，PE，PB，MA10
    reg_factor(factor=['PE', 'PB', 'MA10'])


def on_data(context: Context):
    # 获取注册的因子数据
    data = get_reg_factor(reg_idx=context.reg_factor[0], df=True)
    print(data)

if __name__ == '__main__':
    # 获取上证50的成分股
    target = get_code_list('sz50')
    target = list(target['code'])
    run_backtest(strategy_name='example_test', file_path='.', target_list=target, frequency='day', fre_num=1, begin_date='2018-06-01', end_date='2018-06-30')