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

#%% 导入数据
colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('pima_data.csv', names = colnames)
# 分出特征与标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
from numpy import set_printoptions # 显示精度设置
set_printoptions(precision=3)

#%% 第十五章 自动流程
# 有一些标准的流程可以实现对机器学习问题的自动化处理，
# 在sklearn中通过Pipeline来定义和自动化运行这些流程。

#通过sklearn中的Pipeline实现自动化流程处理。（自动化流程处理的工具）
#  *如何通过Pipeline来最小化数据缺失
#  *如何构建数据准备和生成模型的Pipeline
#  *如何构建特征选择和生成模型的Pipeline

#评估框架中的数据缺失
#训练数据集与评估数据集之间的数据泄露问题

# =============================================================================
# 15.2 数据准备和生成模型的pipeline
#  解决数据泄露问题，需要有一个合适的方式把数据分离成训练集和评估集，这包含于数据准备过程中。
#  Pipeline能够处理训练数据集与评估数据集之间的数据泄露问题，
#  通常会在数据处理过程中对分离出的所有数据子集"做同样的数据处理"。

#如：用pipeline来处理这个过程，分两个步骤：
#（1）正态化数据
#（2）训练一个线性判别分析模型
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline #
kfold = KFold(n_splits= 10, random_state= 7)
steps = [] #--------
steps.append(('Standardize', StandardScaler())) # 添加一个元祖
steps.append(('lda', LinearDiscriminantAnalysis())) #
model = Pipeline(steps) #-------构建这样的模型。。。
result = cross_val_score(model, X, y, cv= kfold)
print('acc_mean = %.3f' % result.mean())

# =============================================================================
# 15.3 特征选择和生成模型的pipeline
# 特征选择也是一个容易受到数据泄露影响的过程。
# 和数据准备一样，特征选择也必须确保数据的稳固性，
# Pipeline也提供了一个工具(FeatureUnion)来保证数据特征选择时数据的稳固性。
# 这个过程包括以下四个步骤：
#  （1）通过主成分分析进行特征选择
#  （2）通过统计选择进行特征选择
#  （3）特征集合
#  （4）生成一个逻辑回归模型
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline # 
from sklearn.pipeline import FeatureUnion #
kfold = KFold(n_splits= 10, random_state= 7)
# 生成 FeatureUnion
features = []
features.append(('pca', PCA()))
features.append(('select_best', SelectKBest(k= 6)))
# 生成 Pipeline
steps = []
steps.append(('feature_union', FeatureUnion(features)))
steps.append(('logistic', LogisticRegression()))
model = Pipeline(steps)
result = cross_val_score(model, X, y, cv= kfold)
print('acc_mean = %.3f' % result.mean())

#%%



#%% 草稿