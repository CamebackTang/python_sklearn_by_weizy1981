# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:15:48 2019

@author: windows
"""
#%% 项目介绍
# Pima Indians数据集：分类问题
# 记录了印第安人最近五年内是否患糖尿病的医疗数据。
#包括：
#第十四章 算法比较（一种模式/一种框架）

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir(r'E:\大学\大学课程\专业课程\机器学习\机器学习python实践')
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

#%% 第十四章 算法比较（一种模式/一种框架）
#一种可以比较不同算法的模式（模板）可视化
from sklearn.model_selection import KFold # K折
from sklearn.model_selection import cross_val_score # 交叉验证

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

# 评估算法
results = []
for key in models: # eee
    kfold = KFold(n_splits=10, random_state= 7)
    cv_results = cross_val_score(models[key], X, y,
            cv= kfold, scoring= 'accuracy') # ndarray
    results.append(cv_results)
    print('{}: {} ({})'.format(key, cv_results.mean(), cv_results.std()))

# 箱线图比较算法(可视化比较)
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)#
ax.set_xticklabels(models.keys())
plt.show()

#%%



#%% 草稿