# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:15:48 2019
本章主要内容：
如何端到端地完成一个分类问题的模型
如何通过（数据转换、调参、集成算法）提高模型的准确度
@author: windows
"""
#%% 项目介绍
#在这个项目中将采用声呐、矿山和岩石数据集
#http://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+%28Sonar%2C+Mines+vs.+Rocks%29
#通过声呐返回的信息判断物质是金属还是岩石。
#数据集共208条数据，每条数据记录了60种不同的声呐探测的数据和一个分类结果
#若结果是岩石则标记为R，否则为金属标记为M

#%% 机器学习项目的python模板
# 通常是六个步骤：
# 1.定义问题
# a）导入类库
# b）导入数据集
# 
#%% 导入类库
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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier

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
dataset = pd.read_csv('sonar.all-data.csv', header=None)


#%% 2.理解数据
print('数据维度：行 %s, 列 %s' %dataset.shape)
#print(dataset.dtypes) # 包含在下info()里面了
print(dataset.info()) #缺失值情况，数据类型
print(dataset.head()) #可以看到：不同单位
# a）描述性统计
print(dataset.groupby(60).size()) # 数据分类的分布情况
print(dataset.describe()) # 对max、min的理解
#print(dataset.corr(method='pearson'))# 大的相关系数
print(dataset.skew()) # 偏离程度
# b）数据可视化
dataset.hist(sharex=False, sharey=False, xlabelsize=1, ylabelsize=1)
plt.show() # 大部分数据呈现高斯分布或指数分布
dataset.plot(kind='density', subplots=True, layout=(8,8),sharex=False,fontsize=1)
plt.show() # 更容易的看出双峰，？指数分布的密度函数长啥样？
'''
大部分数据呈现一定程度的偏态分布，也许通过Box-Cox转换可以提高模型的准确度
Box-Cox转换是统计中常用的一种数据变化方式，用于连续响应变量不满足正态分布的情况
Box-Cox转换后，可以在一定程度上减少不可观测的误差，也可以预测变量的相关性，
将数据转换成正态分布
'''
dataset.plot(kind='box', subplots=True, layout=(8,8), sharex=False,sharey=False, fontsize=1)
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
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
test_size = 0.2
seed = 7
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= test_size, random_state = seed)
# b）定义模型评估标准
#scoring = 'neg_mean_squared_error'
scoring = 'accuracy'

# c）算法审查(评估算法——baseline)
#分析完数据后，并不能决定哪个算法对这个问题最有效。
#直观的感觉是，也行基于距离计算的算法会有不错的表现。

# ======== 原始数据
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()
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
可以看到KNN的结果分布比较紧凑，说明算法对数据的处理比较准确
但SVM算法表现不佳，这出乎我们的意料。
也许是因为数据分布的多样性导致SVM算法不够准确。
下面将对数据进行正态化处理，再次比较算法的结果
'''
# ==========正态化数据
# 为了防止数据泄露，采用Pipeline来正态化数据和对模型进行评估。
#steps = [] #--------
#steps.append(('Scaler', StandardScaler()))
#steps.append(('LR', LinearRegression()))
#pipelines['ScalerLR'] = Pipeline(steps) # model: Scaler+LR
pipelines = {} # models
pipelines['ScalerLR'] = Pipeline([('Scaler',StandardScaler()), ('LR',LogisticRegression())])
pipelines['ScalerLDA'] = Pipeline([('Scaler',StandardScaler()), ('LDA',LinearDiscriminantAnalysis())])
pipelines['ScalerKNN'] = Pipeline([('Scaler',StandardScaler()), ('KNN',KNeighborsClassifier())])
pipelines['ScalerCART'] = Pipeline([('Scaler',StandardScaler()), ('CART',DecisionTreeClassifier())])
pipelines['ScalerNB'] = Pipeline([('Scaler',StandardScaler()), ('NB',GaussianNB())])
pipelines['ScalerSVM'] = Pipeline([('Scaler',StandardScaler()), ('SVM',SVC())])
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
'''可以看到KNN和SVM的数据分布最紧凑'''

#%% 5.优化模型
# a）算法调参
#======== KNN算法调参
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_neighbors': range(1,23,2)}
model = KNeighborsClassifier()
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
#=========SVM算法调参
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {}
param_grid['C'] = np.arange(0.1,2.0,0.2)
#param_grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
param_grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
model = SVC()
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
'''
最好的SVM的参数是C=1.5,kernel=RBF。准确度达到0.8675，高于KNN
'''
    
# b）集成算法
ensembles = {} # models
ensembles['ScalerAB'] = Pipeline([('Scaler',StandardScaler()), ('AB',AdaBoostClassifier())])
ensembles['ScalerGBM'] = Pipeline([('Scaler',StandardScaler()), ('GBM',GradientBoostingClassifier())])
ensembles['ScalerRF'] = Pipeline([('Scaler',StandardScaler()), ('RFR',RandomForestClassifier())])
ensembles['ScalerET'] = Pipeline([('Scaler',StandardScaler()), ('ETR',ExtraTreesClassifier())])
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
model = GradientBoostingClassifier()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator= model, param_grid= param_grid,
                    scoring= scoring, cv= kfold)
grid_result = grid.fit(X=rescaledX, y=y_train)
print('最优: {} 使用{}'.format(grid_result.best_score_,
                              grid_result.best_params_))
#（2） 集成算法ET调参(每次的结果并不一样)   
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = {'n_estimators':[5]+np.linspace(10,100,10).astype(int).tolist()}
model = ExtraTreesClassifier()
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
model = SVC(C=1.5, kernel='rbf')
model.fit(X=rescaledX, y= y_train)
# 评估算法模型
rescaledX_test = scaler.transform(X_test)
predictions = model.predict(rescaledX_test)
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# 6.结果部署
# a）预测评估数据集
# b）利用整个数据集生成模型
# c）序列化模型

#%% 使用模板的小技巧
#（1）快速执行一遍：
#（2）整个流程不是线性的，需要多次重复步骤3和4
#（3）尝试每一个步骤，即使你认为不适用，也不要跳过，而是减少该步骤所做的贡献。

#%% 草稿