#-*-coding:utf-8-*-
#沪指走势预测，使用时间序列ARMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
#数据加载
df=pd.read_csv('./shanghai_1990-12-19_to_2019-2-28.csv')
#将时间作为df的索引
df.Timestamp=pd.to_datetime(df.Timestamp)
df.index=df.Timestamp
#数据探索
print(df.head(5))
#安照月来统计
df_month=df.resample('M',how=np.mean)
#按照月来显示比特币的走势
fig=plt.figure(figsize=[15,7])
plt.rcParams['font.sans-serif']=['SimHei']
plt.suptitle('沪指',fontsize=20)
plt.plot(df_month.Price,'-',label='按月')
plt.legend()
plt.show()
#设置参数范围
ps=range(0,3)
qs=range(0,3)
parameters=product(ps,qs)
parameters_list=list(parameters)
#寻找最优参数使ARMA模型AIC最小
results=[]
best_aic=float('inf')	#正无穷
for param in parameters_list:
	try:
		model=ARMA(df_month.Price,order=(param[0],param[1])).fit()
	except ValueError:
		print('参数错误：',param)
		continue
	aic=model.aic
	if aic<best_aic:
		best_model=model
		best_aic=aic
		best_param=param
	results.append([param,model.aic])
#输出最优模型
results_table=pd.DataFrame(results)
results_table.columns=['parameters','aic']
print('最优模型：',best_model.summary())
#比特币预测
df_month2=df_month[['Price']]
date_list = [datetime(2019, 3, 31), datetime(2019, 4, 30), datetime(2019, 5, 31), datetime(2019, 6, 30), datetime(2019, 7, 31),
             datetime(2019, 8, 31), datetime(2019, 9, 30), datetime(2019, 10, 31),datetime(2019, 11, 30),datetime(2019, 12, 31)]
future=pd.DataFrame(index=date_list,columns=df_month.columns)
df_month2=pd.concat([df_month2,future])
df_month2['forecast']=best_model.predict(start=0,end=348)
#比特币预测结果显示
plt.figure(figsize=(20,7))
df_month2.Price.plot(label='实际指数')
df_month2.forecast.plot(color='r',ls='--',label='预测指数')
plt.legend()
plt.title('沪指（月）')
plt.xlabel('时间')
plt.ylabel('指数')
plt.show()
