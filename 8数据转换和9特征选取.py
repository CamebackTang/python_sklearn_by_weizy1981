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
第八章 数据预处理
第九章 数据特征选定

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

#%% 第八章 数据预处理
# =============================================================================
# # 利用sklearn来转换数据，提供了两种标准的格式化数据的方法
# # (1) 适合和多重变换 (Fit and Multiple Transform)
# # 推荐， fit数据得到变换参数，再transform其他数据
# # (2) 适合和变换组合 (Combined Fit-and-Transform)
# # 对绘图或汇总处理具有非常好的效果
# 
# # ======几种数据转换方法：
# # (1)调整数据尺度 (rescale data)：相同尺度度量
# # (2)正态化数据 (standardize data)：有效处理高斯分布数据
# # (3)标准化数据 (normalize data)：适合处理稀疏数据
# # (4)二值数据 (binarize data)：生成明确值或增加属性 
# # ===========================
# =============================================================================

# 分出特征与标签
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
from numpy import set_printoptions # 显示精度设置
set_printoptions(precision=3)

# (1)调整数据尺度 (rescale data)
# 0~1 梯度下降、回归、神经网络、KNN
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler(feature_range=(0,1))
X_mm_sc = mm_scaler.fit_transform(X) # ndarray
print(X_mm_sc)
# (2)正态化数据 (standardize data)：有效处理高斯分布数据
# 线性回归、逻辑回归、LDA
from sklearn.preprocessing import StandardScaler
sd_scaler = StandardScaler()
X_sd_sc = sd_scaler.fit_transform(X) # ndarray
print(X_sd_sc)
# (3)标准化数据 (normalize data)：适合处理稀疏数据
# 距离变为1(单位化？)。。。"归一元"处理
# 将每一行的数据的距离处理成1的数据？？？
# 使用权重输入的神经网络 和 使用距离的KNN
from sklearn.preprocessing import Normalizer # 范数
nm_scaler = Normalizer(norm='l2')
X_nm_sc = nm_scaler.fit_transform(X)
print(X_nm_sc)
# (4)二值数据 (binarize data) ：生成明确值或增加属性 
# 大于阈值设为1，小于阈值设为0
# 在生成明确值或增加属性的时候使用
from sklearn.preprocessing import Binarizer
bizer = Binarizer(threshold= 0.0)
X_b_sc = bizer.fit_transform(X)
print(X_b_sc)

#%% 第九章 数据特征选定
# 特征选择(4个方法)
# 通过sklearn来自动选择用于机器学习模型的数据特征的方法
# 减少无关的、冗余的，可以提高算法精度及训练时间

#(1)单变量特征选定
# =============================================================================
# 统计分析可以用来分析选择"对结果影响最大"的数据特征
# sklearn中提供了SelectKBest类，可以实现卡方检验
# 卡方检验是检验定性自变量对定性因变量的相关性的方法
# 假设自变量N种取值，应变量M种取值，
# 考虑x=i&y=j的样本频数的观察值与期望值的差距，构建统计量
# 卡方检验就是统计样本实际观察值与理论推断值之间的偏离程度，
# 偏离程度决定卡方值的大小，卡方值越大，偏差越大，越不符合
# 若两个值完全相等，卡方值就为0，表明理论值完全符合。
# =============================================================================
#SelectKBest不仅可以用chi2，还可用相关系数、互信息
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
test = SelectKBest(score_func = chi2, k=4) # 选择4个
fit = test.fit(X,y)
print(fit.scores_) 
features = fit.transform(X) # 取出得分最高的K个属性
print(features)

#(2)递归特征消除(RFE)
# =============================================================================
# 使用一个基模型来进行多轮训练，每轮训练后消除若干权重系数的特征，
# 再基于新的特征集进行下一轮的训练。通过每一个基模型的精度，
# 找到对最终的预测结果影响最大的数据特征。
# =============================================================================
#以逻辑回归算法为基模型，通过RFE来选定对结果影响最大的3个特征
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, y)
print('特征个数：{}'.format(fit.n_features_))
print('被选定的特征：', fit.support_)
print('特征排名', fit.ranking_)

#(3)主成分分析
# =============================================================================
# 常见的数据降维方法有PCA和LDA，二者有很多相似之处，
# 本质是将原始的样本映射到维度更低的样本空间中，但映射目标不一样：
# PCA是为了让映射后的样本具有最大的发散性，
# LDA是为了让映射后的样本有最好的分类性能，本身也是一个分类模型
# LDA是有监督的，PCA是无监督的。
# 在聚类算法中，通常用PCA降维，以利于对数据的简化分析和可视化。
# =============================================================================
from sklearn.decomposition import PCA
pca =  PCA(n_components= 3)
fit = pca.fit(X)
print('解释方差：', fit.explained_variance_ratio_)
print(fit.components_)

#(4)特征的重要性
# =============================================================================
# 这三个算法都是集成算法中的袋装算法
# 袋装决策树算法(Bagged Decision Trees)、RF、极端随机树算法
# 都可以用来计算数据特征的重要性。
# =============================================================================
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
fit = model.fit(X, y)
print('特征得分(重要性)：', fit.feature_importances_)

#%% 草稿

