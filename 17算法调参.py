# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:15:48 2019

@author: windows
"""

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
np.set_printoptions(precision=3)

#%% 导入数据
colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pima_data.csv', names = colnames)
# 分出特征与标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
from numpy import set_printoptions # 显示精度设置
set_printoptions(precision=3)

#%% 
#如何找到最佳的参数组合，可以把它当作一个查询问题来处理，到何时停止？
#（应该遵循偏差和方差协调原则）
#调整算法参数是采用机器学习解决问题的（最后一个步骤），
#有时也被称为（超参数优化）。参数可分两种：
#一种是影响模型在训练集上的（准确度）或（防止过拟合能力）的参数；
#另一种是不影响这两者的参数。模型在样本总体上的准确度有这两者共同决定。
#
#两种自动寻找最优化参数的算法：
#（1）网格搜索优化参数 （网格寻优）
#（2）随机搜索优化参数

#%%（1）网格搜索优化参数 （网格寻优）
#通过（遍历）已定义参数的列表，来评估算法的参数，从而找到最优参数。
#网格搜索优化参数适用于（三四个或更少）的超参数，参数较多时要改用随机搜索。
#当超参数的数量增加时，网格搜索的计算复杂度会呈现（指数型增长）。

#在sklearn中使用GridSearchCV来实现对参数的跟踪、调整与评估，从而找到最优参数。
#GridSearchCV使用（字典对象）来指定需要调参的参数，
#可以同时对一个或多个进行调参。
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
# 算法实例化
model = Ridge()
#设置要遍历的参数(以算法的参数名为key，参数值为列表，可设置多个key-value)
param_grid = {'alpha': [1, 0.1, 0.01, 0.001, 0]}
# 通过网格搜索查询最优参数
grid = GridSearchCV(estimator= model, param_grid= param_grid)
grid.fit(X,y)
# 搜索结果
print('最高得分：%.3f' %grid.best_score_)
print('最优参数：%.3f' %grid.best_estimator_.alpha)
#为什么叫网格？因为只是各参数列表的笛卡尔积？

#%%（2）随机搜索优化参数
# 随机搜索通过（固定次数的迭代），采用随机采样分布的方式搜索合适的参数。
# 随机搜索为每个参数定义了一个分布函数，并在该空间中采样。
# 
# 下面的例子通过RandomizedSearchCV对Ridge算法的参数进行100次迭代，并选出最优的参数。
# Scipy中的uniform是一个均匀随机采样函数，默认生成0与1之间的随机采样数值。
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
# 算法实例化
model = Ridge()
# 设置要遍历的参数
param_grid = {'alpha': uniform}
# 通过随机搜索查询最优参数
grid = RandomizedSearchCV(estimator= model,
                          param_distributions= param_grid,
                          n_iter= 100,
                          random_state= 7)
grid.fit(X,y)
# 搜索结果
print('最高得分：%.3f' %grid.best_score_)
print('最优参数：%.3f' %grid.best_estimator_.alpha)


#%%



#%% 草稿