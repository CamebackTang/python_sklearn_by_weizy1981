# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:15:48 2019

@author: windows
"""
#%% 项目介绍
# Pima Indians数据集：分类问题
# 记录了印第安人最近五年内是否患糖尿病的医疗数据。
包括：
第十二章 审查分类算法
第十三章 审查回归算法

#%%
import numpy as np
import pandas as pd

import os
os.chdir(r'E:\大学\大学课程\专业课程\机器学习\机器学习python实践')
#os.chdir(r'D:\Spyder_Workspace')
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
# 精度要求
pd.set_option('precision',3) 

#%% 导入数据
colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pima_data.csv', names = colnames)
# 分出特征与标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
from numpy import set_printoptions # 显示精度设置
set_printoptions(precision=3)

#%% 第十二章 审查分类算法
#六种机器学习的分类算法
#如何审查机器学习的分类算法
#如何审查两个线性（LR、LDA)分类算法
#如何审查四个非线性(KNN,NB,cart,svm)分类算法
#
#在选择算法时，应该换一种思路，不是针对数据应该采用哪种算法，
#而是应该用数据来审查哪些算法。猜测哪些算法会取得好结果。（数据敏感性）

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#---------
# 逻辑回归
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
kfold = KFold(n_splits= 10, random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%, std= %.3f%%'\
      %(result.mean()*100, result.std()*100))

#---------
# 线性判别分析(LDA)，也叫Fisher线性判别(FLD)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits= 10, random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()*100))

#---------
# K近邻算法（KNN）
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
kfold = KFold(n_splits= 10, random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()*100))

#---------
# 贝叶斯分类器
# 在贝叶斯分类器中，对输入数据同样做了符合高斯分布的假设。
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
kfold = KFold(n_splits= 10, random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()*100))

#---------
# 分类与回归树(CART)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
kfold = KFold(n_splits= 10, random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()*100))

#---------
# 支持向量机（SVM）
# 它在解决小样本、非线性及高维模式识别中表现出许多特有的优势
from sklearn.svm import SVC
model = SVC()
kfold = KFold(n_splits= 10, random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()*100))

#%% 第十三章 审查回归算法
#七种机器学习的分类算法
#如何审查机器学习的回归算法
#如何审查四个线性（LR、ridge、lasso、弹性网络回归)回归算法
#如何审查三个非线性(KNN,cart,svm)回归算法

from sklearn import datasets
boston = datasets.load_boston()
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
X = boston.data
y = boston.target
#---------
# 线性回归
from sklearn.linear_model import LinearRegression
model = LinearRegression()
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits = 10, random_state = 7)
result = cross_val_score(model, X, y, cv= kfold, scoring = scoring)
print('Linear Regression: %.3f' % result.mean())

#-------
# 岭回归
#一种专门用于共线性数据分析的有偏估计回归，实际上是一种改良的最小二乘估计法
#通过放弃最小二乘法的无偏性，以损失部分信息、降低精度为代价，
#获得回归系数更符合实际、更可靠的回归方法，对病态数据的拟合要强于最小二乘法
from sklearn.linear_model import Ridge
model = Ridge()
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits = 10, random_state = 7)
result = cross_val_score(model, X, y, cv= kfold, scoring = scoring)
print('Ridge Regression: %.3f' % result.mean())

#-------
# 套索回归
# 套索回归与岭回归类似，套索回归也会惩罚回归系数，
# 在Lasso中会惩罚回归系数的绝对值大小（在Ridge中惩罚函数是平方？）
# 这导致惩罚（或等于约束估计的绝对值之和）值使得一些参数估计结果等于0。
# 使用惩罚值越大，进一步估计会使缩小值越趋近零。
# 这将导致我们要从给定的n个变量中选择变量。
# 如果预测的一组变量高度相似，lasso会选择其中的一个变量，并将其他的变量收缩为零。
# 此外，它能够减少变化程度并提高线性回归模型的精度。
from sklearn.linear_model import Lasso 
model = Lasso()
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits = 10, random_state = 7)
result = cross_val_score(model, X, y, cv= kfold, scoring = scoring)
print('Lasso Regression: %.3f' % result.mean())

#-------
# 弹性网络回归算法
# 是Lasso回归算法和Ridge回归算法的混合体，
# 在模型训练时，弹性网络回归算法综合使用L1和L2两种正则化方法。
# 当有多个相关的特征时，弹性网络回归算法是很有用的，
# Lasso会随机挑选算法中的一个，而弹性网络则会选两个。
#   与Lasso和Ridge相比，弹性网络的优点是：
#     它允许弹性网络回归继承循环状态下Ridge的一些稳定性。
#     另外，在高度相关变量的情况下，它会产生群体效应；
#     选择变量的数目没有限制；可以承受双重收缩。
from sklearn.linear_model import ElasticNet
model = ElasticNet()
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits = 10, random_state = 7)
result = cross_val_score(model, X, y, cv= kfold, scoring = scoring)
print('ElasticNet Regression: %.3f' % result.mean())

#===============
# K近邻算法（按照距离来预测结果）
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits = 10, random_state = 7)
result = cross_val_score(model, X, y, cv= kfold, scoring = scoring)
print('KNeighbors Regression: %.3f' % result.mean())

#-------
# 分类与回归树
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits = 10, random_state = 7)
result = cross_val_score(model, X, y, cv= kfold, scoring = scoring)
print('CART Regression: %.3f' % result.mean())

#-------
# 支持向量机
from sklearn.svm import SVR
model = SVR()
scoring = 'neg_mean_squared_error'
kfold = KFold(n_splits = 10, random_state = 7)
result = cross_val_score(model, X, y, cv= kfold, scoring = scoring)
print('SVM Regression: %.3f' % result.mean())

#-------

#%% 草稿