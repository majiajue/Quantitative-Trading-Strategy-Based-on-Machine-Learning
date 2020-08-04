# Quantitative-Trading-Strategy-Based-on-Machine-Learning
Firstly, multiple effective factors are discovered through IC value, IR value, and correlation analysis and back-testing. Then, XGBoost classification model is adopted to predict whether the stock is profitable in the next month, and the positions are adjusted monthly. The idea of mean-variance analysis is adopted for risk control, and the volatility of the statistical benchmark index (HS300 Index) is used as a threshold for risk control. Back-testing results: the annual return rate is 11.54%, and the maximum drawdown is 17.91%.

1. Factors Extracting

![image](https://github.com/Yi1214/Quantitative-Trading-Strategy-Based-on-Machine-Learning/blob/master/results/factors_extract_strategy.png)

2. Single Factor Testing

![image](https://github.com/Yi1214/Quantitative-Trading-Strategy-Based-on-Machine-Learning/blob/master/results/single_factor_test_1.png)
![image](https://github.com/Yi1214/Quantitative-Trading-Strategy-Based-on-Machine-Learning/blob/master/results/single_factor_test_2.png)
![image](https://github.com/Yi1214/Quantitative-Trading-Strategy-Based-on-Machine-Learning/blob/master/results/single_factor_test_all.png)

3. XGBOOST Backtesting

![image](https://github.com/Yi1214/Quantitative-Trading-Strategy-Based-on-Machine-Learning/blob/master/results/XGBOOST_Backtesting.png)

4. Out-of-sample Performance

![image](https://github.com/Yi1214/Quantitative-Trading-Strategy-Based-on-Machine-Learning/blob/master/results/XGBOOST_Result1.png)

![image](https://github.com/Yi1214/Quantitative-Trading-Strategy-Based-on-Machine-Learning/blob/master/results/XGBOOST_Result2.png)
