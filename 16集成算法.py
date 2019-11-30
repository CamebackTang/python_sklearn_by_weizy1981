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

#%% 
16.1 集成的方法
三种流行的集成算法的方法
boosting(提升)算法 串性 序列化方法 
 bagging(袋装)算法 并行 并行化方法
  voting(投票)算法 ？？？
bagging：先将训练集分离成多个子集，然后通过各个子集训练多个模型。
boosting：训练多个模型并组成一个序列，序列中的每一个模型都会修正前一个模型的错误。
voting：训练多个模型，并采用样本统计来提高模型的准确度。


#%%
#16.2 袋装算法（bagging)
#通过给定组合投票的方式获得最优解。
#bagging算法在数据具有很大的方差时非常有效
# =============================================================================
# （1）袋装决策树（Bagging Decision Trees), bagging其他也行，不只是树
# 通过BaggingClassifier实现分类与回归树算法
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier #
from sklearn.tree import DecisionTreeClassifier
kfold = KFold(n_splits= 10, random_state= 7)
cart = DecisionTreeClassifier()
num_tree = 100 #100颗树
model = BaggingClassifier(base_estimator= cart, 
                          n_estimators= num_tree,
                          random_state = 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()*100))
# =============================================================================
# （2）随机森林（Random Forest）
# 顾名思义，用随机的方式建立一个森林，由很多决策树组成，树之间没有关联。
# 新样本输入，每棵树判断，哪类多就哪类（涉及结合策略）
#--------
# 在建立每一棵决策树的过程中，有两点需要注意：(采样)与(完全分裂)。
# 首先是(两个随机采样的过程)，RF对输入的数据要进行 行、列的采样。
#--------
# 对于(行采样)采样有放回的方式，得到可能重复的样本集合。
# 假设输入样本有N个，那么采样的样本也为N个。
# 这样在训练时，每棵树的输入样本都不是全部的样本，相对减轻过拟合。------
# 然后进行(列采样)，从M个feature中选出m个（m<<M)。？？？？
# 之后再对采样之后的数据使用(完全分裂的方式)建立决策树,
# 这样决策树的叶子要么(无法再分)，要么所有样本指向同一个类(无需再分)-----
# 这里不用剪枝，因为之前的两个随机采样过程保证了随机性，不剪枝也不会过拟合。
#---------
# 有个比喻：
# 每一棵决策树就是一个精通某一领域的专家，于是对于一个新问题，
# 就可以从不同的角度去看待它，最终由各个专家投票得到结果。
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier #
kfold = KFold(n_splits= 10, random_state= 7)
model = RandomForestClassifier(n_estimators= 100, 
                               random_state= 7,
                               max_features= 3)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()))

# =============================================================================
# （3）极端随机树（Extra Trees） (随机性在于划分属性完全随机选取)
# 与RF十分相似，都是有很多树构成，两个主要区别：
# 1.RF应用的是bagging模型，而ET是使用（所有的训练样本）得到每棵树
# 2.RF是在一个（随机子集）内得到最优分叉特征属性，ET是(完全随机地选择分叉属性)
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
kfold = KFold(n_splits= 10, random_state= 7)
model = ExtraTreesClassifier(n_estimators= 100,
                             random_state= 7,
                             max_features= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()))
# =============================================================================

#%% 提升算法
#提升算法是一种用来提高弱分类算法准确度的方法，该方法先构造一个预测函数系列
#然后以一定的方式将它们组合成一个预测函数。（加强融合）
#---------
#（1）Adaboost (基学习器默认为None ？)
# 针对同一个训练集训练不同的分类器，然后把分类器集合起来。
# 其算法本身是通过（改变数据分布）来实现的，（调整每个样本的权重）
# Adaboost分类器可以排除一些不必要的训练数据特征，并放在关键的训练数据上
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
kfold = KFold(n_splits= 10, random_state= 7)
model = AdaBoostClassifier(n_estimators= 30,
                           random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()))

#--------
#（2）随机梯度提升(Stochastic Gradient Boosting, ? SGB)
# 随机梯度提升法(GBM)基于的思想是：要找到某个函数的最大值，最好的办法就是
# 沿着该函数的梯度方向探寻。（梯度方向，函数值增长最快）
# 由于梯度提升算法在每次更新数据集时都需要遍历整个数据集，计算复杂度高，
# 而SGB一次只用一个样本点来更新回归系数，极大地改善了算法的计算复杂度。
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
kfold = KFold(n_splits= 10, random_state= 7)
model = GradientBoostingClassifier(n_estimators= 100,
                                   random_state= 7)
result = cross_val_score(model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()))
# 梯度提升和梯度下降其实是一回事？ 损失函数

#%% 投票算法Voting
# 投票算法是一个非常简单的多个机器学习算法的集成算法。
# 投票算法是通过创建两个或多个算法模型，利用投票算法将这些算法包装起来。
# 在实际的应用中，可以对每个子模型的预测结果增加权重，以提高准确度。
# 但是，sklearn中不提供加权算法。。。
 
# 在sklearn中实现投票算法。
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier #
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
kfold = KFold(n_splits= 10, random_state= 7)
models = []
models.append(('logistic', LogisticRegression()))
models.append(('cart', DecisionTreeClassifier()))
models.append(('svm', SVC()))
ensemble_model = VotingClassifier(estimators = models)
result = cross_val_score(ensemble_model, X, y, cv= kfold)
print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()))


#%% 草稿