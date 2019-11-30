# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:15:48 2019
/
@author: windows
"""
#%% 项目介绍
# Pima Indians数据集：分类问题
# 记录了印第安人最近五年内是否患糖尿病的医疗数据。
包括：
第十章 评估方法（数据集划分）
第十一章 评价指标（分类和回归）

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #可视化工具

import os
os.chdir(r'E:\大学\大学课程\专业课程\机器学习\机器学习python实践')
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
# 精度要求
pd.set_option('precision',2) 

#%% 导入数据
colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pima_data.csv', names = colnames)

# 分出特征与标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
from numpy import set_printoptions # 显示精度设置
set_printoptions(precision=3)

#%% 第十章 评估算法
#四种分离数据集的方法，分出训练集和评估集
#不知道用哪个方法时，用K折，不知道K取多少，取10

#(1) 分离训练集和评估集（留出法？）
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
seed = 4 # 随机种子设置
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.33, random_state = seed)
model = LogisticRegression()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print('算法评估结果：%.3f%%' %(result*100))

#(2) K折交叉验证分离
# K折交叉验证是用来评估机器学习算法的黄金法则
# K折可以有效避免过拟合和欠拟合，K通常取3、5、10
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
model = LogisticRegression()
kfold = KFold(n_splits= 10, random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果：mean= %.3f%%, std= %.3f%%'\
      %(result.mean()*100, result.std()*100))

#(3) 弃一交叉验证分离（K取样本数N）(留一法LOO)
#优点1：每一回合几乎所有的样本用于训练，最接近原始样本的分布
#优点2：(实验过程)中没有随机因素会影响实验数据，是可以被复制的
#计算成本高，除非训练快或者可并行
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
model = LogisticRegression()
loocv = LeaveOneOut()
result = cross_val_score(model, X, y, cv= loocv)
print('算法评估结果：mean= %.3f%%, std= %.3f%%'\
      %(result.mean()*100, result.std()*100))
#算法评估结果：mean= 76.953%, std= 42.113%

#(4) 重复随机分离评估、训练集分离
#另一种K折，但是重复这个过程多次（留出法运行10次?）
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
model = LogisticRegression()
skfold = ShuffleSplit(n_splits= 10, test_size= 0.33, random_state = 7)
result = cross_val_score(model, X, y, cv= skfold)
print('算法评估结果：mean= %.3f%%, std= %.3f%%'\
      %(result.mean()*100, result.std()*100))

#%% 第十一章 算法评估指标 (衡量指标选取)
#分类算法评价指标 以逻辑回归为例
#回归算法评价指标 以线性回归为例

# =============================================================================
# 11.2 几种用来评估分类算法的评估指标
#---------
#分类算法评价指标以逻辑回归为例,并选择10折交叉验证
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
model = LogisticRegression()
kfold = KFold(n_splits= 10, random_state= 7)
# (1)分类准确度
#这是分类算法中最常见，也最易被误用的评估参数（类别不平衡）
# scoring = 'accuracy'
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%, std= %.3f%%'\
      %(result.mean()*100, result.std()*100))
'''算法评估结果(准确度acc)：mean= 76.951%, std= 4.841%'''

# (2)对数损失函数（Logloss）
# 在逻辑回归的推导中，它假设样本服从伯努利分布（0~1分布），
#然后求得满足该分布的似然函数，再取对数、求极值等。
#而逻辑回归并没有求似然函数的极值，而是把极大化当作一种思想，
#进而推导出它的经验风险函数为：最小化负的似然函数
#[max F(y, f(x)) -> min -F(y, f(x))]。
#从损失函数的视角来看，它就成了对数损失函数了。
#损失越小，模型越好，而且使损失函数尽量是一个凸函数，便于收敛计算
scoring = 'neg_log_loss'
result = cross_val_score(model, X, y, cv= kfold, \
                         scoring= scoring)#
print('算法评估结果(Logloss)：mean= %.3f%%, std= %.3f%%'\
      %(result.mean()*100, result.std()*100))

# (3)AUC图
#AUC是ROC曲线下的面积（Area Under ROC Curve）
#ROC是受试者工作特征曲线，又叫感受性曲线(线上的点反映相同的感受性)
#ROC是反映敏感性和特异性连续变量的综合指标，图揭示了他们的相互关系
#敏感性为纵，特异性为横
scoring = 'roc_auc'
result = cross_val_score(model, X, y, cv= kfold, \
                         scoring= scoring)#
print('算法评估结果(Logloss)：mean= %.3f, std= %.3f'\
      %(result.mean(), result.std()))

#------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.33, random_state = 4)
model = LogisticRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test) #
# (4)混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted) #
print(cm)
print(pd.DataFrame(cm, index=[0,1], columns=[0,1]))#
# (5)分类报告（Classification Report）
# 能给出P、R、f1、support，accuracy、macro avg、weighted avg
from sklearn.metrics import classification_report
report = classification_report(y_test, predicted)
print(report)

# =============================================================================
#11.3 几种用来评估分类算法的评估矩阵
#回归算法评价指标 以线性回归为例，波士顿房价预测
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import datasets
boston = datasets.load_boston()
kfold = KFold(n_splits=10, random_state=7)
model = LinearRegression()
#----------
# (1)评价绝对误差(MAE)
scoring = 'neg_mean_absolute_error'
result = cross_val_score(model, boston.data, boston.target, cv=kfold, scoring = scoring)
print('MAE: %.3f (%.3f)' %(result.mean(), result.std()))  
# (2)均方误差(MSE)
scoring = 'neg_mean_squared_error'
result = cross_val_score(model, boston.data, boston.target, cv=kfold, scoring = scoring)
print('MSE: %.3f (%.3f)' %(result.mean(), result.std())) 
# (3)R的平方（决定系数）
#决定系数，反映因变量的全部变异能通过回归关系被自变量解释的比例
# r2=0.8,表示如果我们能控制自变量不变，则因变量的变异程度会减少80%
#拟合优度越大，解释程度越高，自变量引起的变动占总变动的百分比越高
# 是个统计量，需要进行检验
scoring = 'r2'
result = cross_val_score(model, boston.data, boston.target, cv=kfold, scoring = scoring)
print('R2: %.3f (%.3f)' %(result.mean(), result.std())) 

