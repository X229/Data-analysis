#-*-coding:utf-8-*-
#信用卡违约率分析
import pandas as pd
from sklearn.model_selection import learning_curve,train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

#屏蔽警告
warnings.filterwarnings("ignore")
#数据加载
data=pd.read_csv('./UCI_Credit_Card.csv')
#数据探索
print(data.shape)
print(data.describe())
print(data.info())
print(data.columns)
#查看下一个月违约率情况
next_month=data['default.payment.next.month'].value_counts()
print(next_month)
df=pd.DataFrame({'default.payment.next.month':next_month.index,'人数':next_month.values})
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure(figsize=(6,6))
plt.title('信用卡违约用户')
sns.set_color_codes('pastel')
sns.barplot(x='default.payment.next.month',y='人数',data=df)
locs,labels=plt.xticks((0,1),('守约','违约'))
plt.show()

#特征选取，去掉ID和违约结果
data.drop(['ID'],inplace=True,axis=1)
target=data['default.payment.next.month'].values
columns=data.columns.tolist()
columns.remove('default.payment.next.month')
features=data[columns].values
#30%作为测试集,其余作为训练集
train_x,test_x,train_y,test_y=train_test_split(features,target,test_size=0.3,stratify=target,random_state=1)

#构造分类器
classifiers=[
	SVC(random_state=1,kernel='rbf'),
	DecisionTreeClassifier(random_state=1,criterion='gini'),
	RandomForestClassifier(random_state=1,criterion='gini'),
	KNeighborsClassifier(metric='minkowski'),
	AdaBoostClassifier(random_state=1)
]
#分类器名
classifiers_names=['svc',
				  'decisiontreeclassifier',
				  'randomforestclassifier',
				  'kneighborsclassifier',
				  'adaboostclassifier'
]
#分类器参数
classifiers_param_grid=[
	{'svc__C':[1],'svc__gamma':[0.01]},
	{'decisiontreeclassifier__max_depth':[6,9,11]},
	{'randomforestclassifier__n_estimators':[3,4,6]},
	{'kneighborsclassifier__n_neighbors':[4,6,8]},
	{'adaboostclassifier__n_estimators':[10,50,100]}
]

#对各分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,param_grid,score='accuracy'):
	response={}
	gridseach=GridSearchCV(estimator=pipeline,param_grid=param_grid,scoring=score)
	#寻找最优参数和最优准确率分数
	search=gridseach.fit(train_x,train_y)
	print("GridSearchCV最优参数：",search.best_params_)
	print("GridSearchCV最优分数：%.4f"%search.best_score_)
	predict_y=gridseach.predict(test_x)
	print("准确率为:%.4f"%accuracy_score(test_y,predict_y))
	response['predict_y']=predict_y
	response['accuray_score']=accuracy_score(test_y,predict_y)
	return response

for model,model_name,model_param_grid in zip(classifiers,classifiers_names,classifiers_param_grid):
	pipeline=Pipeline([
		('scaler',StandardScaler()),
		(model_name,model)
	])
	result=GridSearchCV_work(pipeline,train_x,train_y,test_x,test_y,model_param_grid,score='accuracy')