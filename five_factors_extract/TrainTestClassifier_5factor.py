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
import pickle
from sklearn.metrics import accuracy_score

# 设置label的界限 ret_class>0.03标记为1，否则为0
def PriceProcess(price_path, ret_class):

    """处理price，得到每个月的ret，并打好标签（）,返回df"""
    price = pd.read_csv(price_path)
    price = price[price['close'] != 0]
    # 在相同的target_idx内计算ret
    price['ret_month'] = price.groupby('target_idx')['close'].apply(lambda x: (x - x.shift()) / x.shift())
    price.loc[price['ret_month'] >= ret_class, 'label'] = 1  # 盈利率>3%, 标记为1
    price.loc[price['ret_month'] < ret_class, 'label'] = 0  # 盈利率<3%, 标记为0
    price_month1 = price[['target_idx', 'time', 'ret_month', 'close', 'label']]
    # 增加month列，201701，201702，只记录月份，不记录日时分秒
    price_month1['month'] = price_month1['time'].apply(lambda x: int(str(x)[0:4] + str(x)[5:7])).copy()
    price_month1['ret_nextmonth'] = price_month1.groupby('target_idx')['ret_month'].shift(-1).copy()  # 添加下个月的盈利
    # 对应平移标签，因为ret_month计算的是本月的盈利
    price_month1['label'] = price_month1.groupby('target_idx')['label'].shift(-1).copy()
    return price_month1

def FactorProcess(factor_path):

    """处理factor,返回df"""
    factor = pd.read_csv(factor_path)
    factor = factor.dropna(subset=['date'])  # 删除非法日期
    #factor['code'] = factor['target_idx'].apply(lambda x: context.target_list[x])  # 将用0，1，2，3等表示的股票换成对应的股票代码
    # 增加month列，2017-01，2017-02，只记录月份，不记录日时分秒
    factor['month'] = factor['date'].apply(lambda x: int(str(x)[0:4] + str(x)[5:7]))
    factor_name = factor['factor'].drop_duplicates().tolist()  # 以列表的形式取出因子名称
    # 将factor按['target_idx','month','factor']分组，分别取每组的最后一行
    # 即取出各股票每个月末的所有因子值
    factor_month = factor.groupby(['target_idx', 'month', 'factor']).apply(lambda x: x.iloc[-1])[
        ['date', 'value']].reset_index()
    # 添加所有因子名作为新的列
    factor_month1 = factor_month.groupby(['target_idx', 'month']).apply(deal).reset_index()
    return factor_month1, factor_name


def deal(df):
    factor_name = df['factor'].tolist()
    df1 = pd.DataFrame(columns=factor_name)
    for i in factor_name:
        df1[i] = df.loc[df['factor'] == i, 'value'].values
    return df1

def ObtainDataset(price_month1, factor_month1, factor_name, test_num):   # test_num 为需要的交叉验证集的月数
    df = pd.merge(factor_month1, price_month1, on=['target_idx', 'month'], how='right')
    """
    训练集取最初到最后的前两个月，[:-2]  ==>参数化后: [:-1*test_num-1]
    测试集取倒数第二个月[-2：-1]，因为最后一个月实际是没有标签的  ==>参数化后: [-1*test_num-1:-1]
    """
    train = df.groupby('target_idx').apply(lambda x:x.iloc[:-1*test_num-1])
    test = df.groupby('target_idx').apply(lambda x:x.iloc[-1*test_num-1:-1])
    scaler = StandardScaler()   # 标准化

    # 提取因子值train[factor_name]作为特征
    X_train = train[factor_name]
    X_train = X_train.fillna(0).values
    X_train = scaler.fit_transform(X_train)    # 因子标准化
    # 提取train['label']作为标签，
    # .fillna(0)NaN补0，
    # .values转化成array形式
    y_train = train['label'].fillna(0).values

    X_test = test[factor_name]
    X_test = X_test.fillna(0).values
    X_test = scaler.transform(X_test)  # 因子标准化
    y_test = test['label'].values

    return X_train, y_train, X_test, y_test

def TrainModel(price_path, factor_path, ret_class, test_num):
    # obtain dataset
    price_month1 = PriceProcess(price_path, ret_class)
    factor_month1, factor_name = FactorProcess(factor_path)
    X_train, y_train, X_test, y_test = ObtainDataset(price_month1, factor_month1, factor_name, test_num)

    # train model
    model = XGBClassifier().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred)
    # predictions = [round(value) for value in y_pred]

    # save model
    pickle.dump(model, open("XGboost_ret0.03_5factor.pickle.dat", "wb"))
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return y_pred, y_test

if __name__ == "__main__":

    price_path = "price_5factor_test.csv"
    factor_path = "5factor.csv"
    warnings.filterwarnings("ignore")
    y_pred, y_test = TrainModel(price_path, factor_path, ret_class = 0.03, test_num = 12)

