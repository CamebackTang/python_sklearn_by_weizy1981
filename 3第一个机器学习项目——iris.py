# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 18:29:04 2019
第三章：第一个机器学习项目 iris数据集
@author: windows
"""
#%% 项目介绍（分类问题）
#Iris也称鸢尾花卉数据集，是一类多重变量分析的数据集。
#通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个属性，
#预测鸢尾花卉属于（Setosa，Versicolour，Virginica）三个种类中的哪一类。

#%% 机器学习项目的python模板
# 通常是六个步骤：
# 1.定义问题
# a）导入类库
# b）导入数据集

#%% 导入类库，初始化环境设置
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #可视化工具
from pandas.plotting import scatter_matrix # 散点图矩阵绘制
# 模型选择
from sklearn.model_selection import train_test_split # 留出法
from sklearn.model_selection import KFold # K折
from sklearn.model_selection import cross_val_score # 交叉验证
# 衡量指标
from sklearn.metrics import classification_report # 分类评价指标报告
from sklearn.metrics import confusion_matrix # 混淆矩阵
from sklearn.metrics import accuracy_score
# 分类模型
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import os
os.chdir(r'E:\大学\大学课程\专业课程\机器学习\机器学习python实践')
#os.getcwd()
#os.chdir(r'D:\Spyder_Workspace')
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
pd.set_option('display.width',120)
# 精度要求
pd.set_option('precision',3) 
np.set_printoptions(precision=3)

#%% 导入数据（也可以去UCI机器学习仓库下载数据集）
#from sklearn import datasets
#iris = datasets.load_iris()
##iris.data
##iris.feature_names
##iris.target
##iris.target_names
#filename = r'E:\大学\大学课程\专业课程\机器学习\机器学习python实践\iris.csv'
filename = 'iris.csv'
col_names = ['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)',
 'target']
dataset = pd.read_csv(filename, header=None, names = col_names, engine='python')

#%% 概述数据
#(1) 数据的维度
print('数据维度：行 %s, 列 %s' %dataset.shape)
#(2) 查看数据本身
#print(dataset.dtypes) # 包含在下info()里面了
print(dataset.info()) #缺失值情况，数据类型
print(dataset.head(4)) #可以看到：不同单位
#(3) 统计描述所有的数据特征
print(dataset.describe()) # 对max、min的理解
print(dataset.corr(method='pearson'))# 大的相关系数
print(dataset.skew()) # 偏离程度
#(4) 数据分类的分布情况 (检查分类是否均衡)-----!!!
dataset.groupby('target').size()
# 过抽样：复制少数类。 欠抽样：删除多数类

#%% 数据可视化
#数据特征的分布——单变量图表：更好地理解每一个特征
#------箱型图
dataset.plot(kind='box', subplots=True, layout=(2,2), figsize=(7,7), sharex=False, sharey=False)
plt.show()
#------直方图
dataset.hist(figsize=(7,7))
plt.show()

#不同特征之间的相互关系（相关性）——多变量图表
#------散点图矩阵
scatter_matrix(dataset, figsize=(7,7))
plt.show()

#%% 评估算法
# 通过不同的算法来创建模型，并评估，以便选取最合适的算法。
#(1)分离出评估数据集
# 80%的训练集，20%的评估集，用于评估算法模型 --------!!!
seed = 7
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, -1]
X_train, X_validation, Y_train, Y_validation = \
     train_test_split(X, Y, test_size = 0.2, 
                      random_state= seed)
# =============================================================================
# seed = 7 # 同一个随机种子
# from collections import Counter
# X = dataset.iloc[:, 0:4]
# Y = dataset.iloc[:, 4]
# X_train, X_validation, Y_train, Y_validation = \
#     train_test_split(X, Y, test_size = 0.2, 
#                      random_state= seed, stratify=Y) # 20%,分层抽样
# print(Counter(Y_train), Counter(Y_validation))
# =============================================================================

#(2)评估模式
# 10折交叉验证来分离 "训练集"，留一份来评估算法-----!!!
# 使用相同的数据，对每一种算法进行训练和评估，从中选择最好的模型

#(3)创建模型
# ----线性算法
# 线性回归（LR）
# 线性判别分析（LDA）
# -----非线性算法
# K近邻（KNN)
# 分类与回归决策树(CART)
# 贝叶斯分类器（NB）
# 支持向量机（SVM）

#(4)选择最优模型

## 算法审查
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
    kfold = KFold(n_splits=10, random_state= seed)
    cv_results = cross_val_score(models[key], X_train, Y_train,\
            cv= kfold, scoring= 'accuracy') # ndarray
    results.append(cv_results)
    print('{}: {} ({})'.format(key, cv_results.mean(), cv_results.std()))

# 箱线图比较算法
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)#
ax.set_xticklabels(models.keys())
plt.show()
'''
LR: 0.9666666666666666 (0.04082482904638632)
LDA: 0.975 (0.03818813079129868)
KNN: 0.9833333333333332 (0.03333333333333335)
CART: 0.975 (0.03818813079129868)
NB: 0.975 (0.053359368645273735)
SVM: 0.9916666666666666 (0.025000000000000012)
可以看出SVM算法具有最高的准确度得分。
'''

#%% 实施预测
# 从评估结果看出，SVM是准确度最高的算法
# 现在使用预留的"评估数据集"来验证这个算法模型
#!! 现在使用全部训练集的数据生成SVM的算法模型-------!!!

# 使用评估集评估算法
svm = SVC()
svm.fit(X= X_train, y= Y_train)
print(svm.score(X_validation, Y_validation))
predictions = svm.predict(X_validation)
accuracy_score(Y_validation, predictions)
confusion_matrix(Y_validation, predictions)
print(classification_report(Y_validation, predictions))
# 用print好看多了！

#%% 草稿
