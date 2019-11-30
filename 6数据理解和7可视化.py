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
第六章 数据理解：七种方法理解数据
第七章 数据可视化


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

#%% 第六章 数据理解：七种方法理解数据
#（1）简单地查看数据
data.head(10) #--查看前10行数据
data.info() #--查看info()，是否有缺失
#（2）维度
data.shape
#（3）属性和类型
data.dtypes
#（4）分类的分布情况 ------!!!
data.groupby('class').size()
#（5) 描述性统计
data.describe()
#（6）理解数据属性的相关性
# 相关性较高时，有些算法(LR,线性)的性能会下降
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
fig=plt.gcf()
fig.set_size_inches(12,10)
plt.show()
#（7）数据的分布状况
# df.skew()方法来计算属性的高斯分布偏离情况, 接近0好
data.skew() # ----!!!

#%% 第七章 数据可视化
figsize=(7,7)
#-------单一图表
# 直方图
data.hist(figsize= figsize) # 能看得出来哪些是指数分布、正态分布吗？
plt.show()
# 密度图，显示分布，比直方图更直观
data.plot(kind='density', subplots=True, layout=(3,3),figsize=figsize, sharex=False)
# 箱型图
data.plot(kind='box', subplots=True, layout=(3,3),figsize=figsize, sharex=False)

#-------多重图表
# 相关系数矩阵图
corr = data.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr, vmin= -1, vmax= 1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(colnames)
ax.set_yticklabels(colnames)
plt.show()
# 散点图矩阵
from pandas.plotting import scatter_matrix
scatter_matrix(data, figsize=figsize)
plt.show()

# %% 草稿