# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:15:48 2019
本章主要内容：
如何端到端地完成一个回归问题的模型
如何通过（数据转换、调参、集成算法）提高模型的准确度

@author: windows
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #可视化工具
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor 
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

import os
os.chdir(r'E:\大学\大学课程\专业课程\机器学习\机器学习python实践')
#os.chdir(r'D:\Spyder_Workspace')
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
pd.set_option('display.width',120)
# 精度要求
pd.set_option('precision',3) 
np.set_printoptions(precision=3)

#%% 导入数据
from sklearn import datasets
boston = datasets.load_boston()
dataset = pd.DataFrame(boston.data, columns= boston.feature_names)

#%% Boston house predict
#一个很好的实践机器学习项目的方法，
#是使用从UCI机器学习仓库获取的数据集开启一个机器学习项目。
#%% 机器学习项目的python模板
# 通常是六个步骤：
# 1.定义问题
# a）导入类库
# b）导入数据集
# 
#%% 2.理解数据
print('数据维度：行 %s, 列 %s' %dataset.shape)
#print(dataset.dtypes) # 包含在下info()里面了
print(dataset.info()) #缺失值情况，数据类型
print(dataset.head()) #可以看到：不同单位
# a）描述性统计
print(dataset.describe()) # 对max、min的理解
print(dataset.corr(method='pearson'))# 大的相关系数
print(dataset.skew()) # 偏离程度
# b）数据可视化
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show() # 有指数分布、双峰分布
dataset.plot(kind='density', subplots=True, layout=(4,4),sharex=False,fontsize=1)
plt.show() # 更容易的看出双峰，？指数分布的密度函数长啥样？
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False,sharey=False, fontsize=1)
plt.show() # 偏离程度
scatter_matrix(dataset)
plt.show()
sns.heatmap(dataset.corr(), annot=True, cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.show()# 强相关的，建议后续处理中移除这些特征。。。
'''通过数据的相关性和分布等发现，数据集中数据结构比较复杂
需要考虑对数据进行转换，以提高准确度
1.通过特征选择来减少大部分相关性高的特征。
2.通过标准化数据来降低不同数据度量单位带来的影响。
3.通过正态化数据来降低不同的数据分布结构，以提高准确度
可进一步查看数据的可能性分级（离散化），他可以提高决策树准确度
'''

#%% 3.数据准备
# a）数据清洗
# b）特征选择
# c）数据转换

#%%  4.评估算法（评估框架来选择合适的算法）
# a）分离数据集
X = dataset.values
y = boston.target
#y = pd.DataFrame(boston.target,columns=['MEDV'])
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.2, random_state = 7)
# b）定义模型评估标准
scoring = 'neg_mean_squared_error'
# c）算法审查(评估算法——baseline)
# ======== 原始数据
models = {}
models['LR'] = LinearRegression()
models['LASSO'] = Lasso()
models['EN'] = ElasticNet()
models['KNN'] = KNeighborsRegressor()
models['CART'] = DecisionTreeRegressor()
models['SVM'] = SVR()
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_result = cross_val_score(models[key], X_train,y_train,
                                cv=kfold, scoring= scoring)
    results.append(cv_result)
    print('{}: {} {}'.format(key, cv_result.mean(), cv_result.std()))
# d）算法比较
# 箱线图
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()
'''
不同的数据度量单位，也许是KNN和SVM算法表现不佳的主要原因吧
下面将对数据进行正态化处理，再次比较算法的结果
'''
# ==========正态化数据
# 为了防止数据泄露，采用Pipeline来正态化数据和对模型进行评估。
#steps = [] #--------
#steps.append(('Scaler', StandardScaler()))
#steps.append(('LR', LinearRegression()))
#pipelines['ScalerLR'] = Pipeline(steps) # model: Scaler+LR
pipelines = {} # models
pipelines['ScalerLR'] = Pipeline([('Scaler',StandardScaler()), ('LR',LinearRegression())])
pipelines['ScalerLASSO'] = Pipeline([('Scaler',StandardScaler()), ('LASSO',Lasso())])
pipelines['ScalerEN'] = Pipeline([('Scaler',StandardScaler()), ('EN',ElasticNet())])
pipelines['ScalerKNN'] = Pipeline([('Scaler',StandardScaler()), ('KNN',KNeighborsRegressor())])
pipelines['ScalerCART'] = Pipeline([('Scaler',StandardScaler()), ('CART',DecisionTreeRegressor())])
pipelines['ScalerSVM'] = Pipeline([('Scaler',StandardScaler()), ('SVM',SVR())])
results = []
for key in pipelines:
    kfold = KFold(n_splits=10, random_state=7)
    cv_result = cross_val_score(pipelines[key], X_train,y_train,
                                cv=kfold, scoring= scoring)
    results.append(cv_result)
    print('{}: {} {}'.format(key, cv_result.mean(), cv_result.std()))
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(pipelines.keys())
plt.show()
'''可以看到KNN具有最优的MSE和最紧凑的数据分布'''

#%% 5.优化模型
# a）算法调参
# 网格搜索KNN的K。
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_neighbors': range(1,23,2)}
model = KNeighborsRegressor()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator= model, param_grid = param_grid,
                    scoring = scoring, cv = kfold)
grid_result = grid.fit(X=rescaledX, y=y_train)

print('最优: {} 使用{}'.format(grid_result.best_score_,
                              grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('{} ({}) with {}'.format(mean, std, param))

# b）集成算法
ensembles = {} # models
ensembles['ScalerAB'] = Pipeline([('Scaler',StandardScaler()), ('AB',AdaBoostRegressor())])
ensembles['ScalerAB-KNN'] = Pipeline([('Scaler',StandardScaler()), ('ABKNN',AdaBoostRegressor(
        base_estimator= KNeighborsRegressor(n_neighbors=3)))])
ensembles['ScalerAB-LR'] = Pipeline([('Scaler',StandardScaler()), ('ABLR',AdaBoostRegressor(
        base_estimator= LinearRegression()))])
ensembles['ScalerRFR'] = Pipeline([('Scaler',StandardScaler()), ('RFR',RandomForestRegressor())])
ensembles['ScalerETR'] = Pipeline([('Scaler',StandardScaler()), ('ETR',ExtraTreesRegressor())])
ensembles['ScalerGBR'] = Pipeline([('Scaler',StandardScaler()), ('GBR',GradientBoostingRegressor())])
results = []
for key in ensembles:
    kfold = KFold(n_splits=10, random_state=7)
    cv_result = cross_val_score(ensembles[key], X_train,y_train,
                                cv=kfold, scoring= scoring)
    results.append(cv_result)
    print('{}: {} {}'.format(key, cv_result.mean(), cv_result.std()))
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembles.keys())
plt.show()
    
#20.9 集成算法调参！！！
#集成算法都有一个参数n_estimators，这是一个很好的可以用来调整的参数。
#下面对随机梯度上升（GBM）和极端随机树（ET）算法进行调参，来确定最终的模型。
#（1） 集成算法GBM调参(每次的结果并不一样，为什么？)   
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators':[10,50,100,200,300,400,500,600,700,800,900]}
model = GradientBoostingRegressor()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator= model, param_grid= param_grid,
                    scoring= scoring, cv= kfold)
grid_result = grid.fit(X=rescaledX, y=y_train)
print('最优: {} 使用{}'.format(grid_result.best_score_,
                              grid_result.best_params_))
#（1） 集成算法ET调参(每次的结果并不一样)   
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators':[5]+np.linspace(10,100,10).astype(int).tolist()}
model = ExtraTreesRegressor()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator= model, param_grid= param_grid,
                    scoring= scoring, cv= kfold)
grid_result = grid.fit(X=rescaledX, y=y_train)
print('最优: {} 使用{}'.format(grid_result.best_score_,
                              grid_result.best_params_))


#20.10 确定最终模型
# 训练模型
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
gbr = ExtraTreesRegressor(n_estimators=80)
gbr.fit(X=rescaledX, y= y_train)
# 评估算法模型
rescaledX_test = scaler.transform(X_test)
predictions = gbr.predict(rescaledX_test)
print(mean_squared_error(y_test, predictions))


# 6.结果部署
# a）预测评估数据集
# b）利用整个数据集生成模型
# c）序列化模型

#%% 使用模板的小技巧
#（1）快速执行一遍：
#（2）整个流程不是线性的，需要多次重复步骤3和4
#（3）尝试每一个步骤，即使你认为不适用，也不要跳过，而是减少该步骤所做的贡献。

#%% 草稿