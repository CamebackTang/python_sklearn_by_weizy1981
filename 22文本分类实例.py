# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:15:48 2019
本章主要内容：
如何端到端地完成一个文本分类问题的模型
如何通过文本特征提取生成数据特征
如何通过（调参、集成算法）提高模型的准确度
@author: windows
"""
#%% 项目介绍
#在这个项目中将采用20Newgroups的数据（http://qwone.com/~jason/20Newgroups/）
#这个是网上非常流行的对文本进行分类和聚类的数据集。
#数据集分两部分，训练数据和评估数据。
#网上还提供了3个数据集，这里采用20news-bydata这个数据集进行项目研究。
#这个数据集是按照日期进行排序的，并去掉了部分重复数据和header，共包含18846个文档

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
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB# 这次不是GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer #文本特征提取：
from sklearn.feature_extraction.text import TfidfVectorizer

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

#%% 导入数据
#这里使用scikit-learn的loadfile导入文档数据，
#文档是按照不同的分类分目录来保存的，文件目录名称即所属类别。
categories = ['alt.atheism',
              'rec.sport.hockey',
              'comp.graphics',
              'sci.crypt',
              'comp.os.ms-windows.misc',
              'sci.electronics',
              'comp.sys.ibm.pc.hardware',
              'sci.med',
              'comp.sys.mac.hardware',
              'sci.space',
              'comp.windows.x',
              'soc.religion.christian',
              'misc.forsale',
              'talk.politics.guns',
              'rec.autos' 
              'talk.politics.mideast',
              'rec.motorcycles',
              'talk.politics.misc',
              'rec.sport.baseball',
              'talk.religion.misc']

#导入训练数据和评估数据
from sklearn.datasets import load_files
train_path = '20news-bydate-train'
test_path = '20news-bydate-test'
dataset_train = load_files(container_path= train_path, categories= categories)
dataset_test = load_files(container_path= test_path, categories= categories)

#%% 22.3 文本特征提取
# =============================================================================
# 文本数据属于非结构化的数据，一般要转换成结构化的数据才能进行ML文本分类。
# 常见的做法是将文本转换成“文档-词项矩阵”，
# 矩阵的元素可以用词频或TF-IDF值等。TF-IDF实际上是TF*IDF。
# TF-IDF值是一种用于信息检索与数据挖掘的常用加权技术。
# TF的意思是词频（Term Frequency），IDF是逆向文件频率（Inverse Document Frequency）。
# 
# TF(t)= 该词语在当前文档出现的次数 / 当前文档中词语的总数
# IDF(t)= log_e（文档总数 / 出现该词语的文档总数）
#
# TF-IDF的主要思想是：如果某一个词或短语在一篇文章中出现的频率高，
# 并且在其他文章中很少出现，则认为此词或短语具有很好的类别区分能力，适合用来分类。
# 
# IDF的主要思想是：如果包含词条t的文档数n越小，IDF越大，类别区分能力好。
# 如果某一类文档C中包含词条t的文档数为m，而其他类包含词条t的总数为k，
# 显然n=m+k，当m大的时候，n也大，按照IDF公式得到的IDF值小，说明t的区分能力不强。
# 
# 但实际上，如果一个词条在一个类的文档中频繁出现，
# 则说明该词条能够很好地代表这个类的文本特征，这样的词条应该被赋予较高的权重，
# 并作为该类文本的特征词，以区别于其他类文档。
# 
# 这就是IDF的不足之处，在一份给定的文件里，
# TF指的是某一个给定的词语在该文件中出现的频率，
# 这是对词数（Term Count)的归一化，以防止它偏向长的文件。
# 
# IDF是一个词语普遍重要性的度量，
# 某一个特定词语的IDF，可以由总文件数目除以包含该词语的文件的数目，再取对数。
# =============================================================================

#scikit-learn中提供了词频和TF-IDF来进行文本特征提取的实现，
#分别是CountVectorizer和TfidfTransformer.
# 计算词频
count_vect = CountVectorizer(stop_words= 'english', decode_error= 'ignore')
X_train_counts = count_vect.fit_transform(dataset_train.data)
print(X_train_counts.shape)
# 计算TF-IDF
tf_transformer = TfidfVectorizer(stop_words= 'english', decode_error= 'ignore')
X_train_counts_tf = tf_transformer.fit_transform(dataset_train.data)
print(X_train_counts_tf.shape)
'''
这里通过两种方法进行了文本特征的提取，并且查看了数据维度，得到的维度非常大。
在后续的项目中，将使用TF-IDF进行分类模型的训练。
因为，TF=IDF的数据维度巨大，并且自用提取的特征数据，进一步对数据进行分析的意义不大，
因此只简单地查看数据维度的信息。（没必要进行过多的数据理解？）
'''

#%% 2.理解数据
# a）描述性统计
# b）数据可视化

#%% 3.数据准备
# a）数据清洗
# b）特征选择
# c）数据转换

#%%  4.评估算法（评估框架来选择合适的算法）
# a）分离数据集

# b）定义模型评估标准
#scoring = 'neg_mean_squared_error'
scoring = 'accuracy'

# c）算法审查(评估算法——baseline)
#分析完数据后，并不能决定哪个算法对这个问题最有效。

# ======== 
models = {}
models['LR'] = LogisticRegression()
#models['SVM'] = SVC()
models['CART'] = DecisionTreeClassifier()
models['MNB'] = MultinomialNB()
models['KNN'] = KNeighborsClassifier()
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_result = cross_val_score(models[key], 
                   X_train_counts_tf, dataset_train.target,
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
结果显示：逻辑回归具有最好的准确度，NB和KNN值得进一步研究。
从图可得：NB的数据离散程度比较好，逻辑回归的偏度较大
'''

#%% 5.优化模型
# a）算法调参
#======== 逻辑回归算法调参
#在逻辑回归中的超参数是C。C是目标的约束函数，C值越小则正则化强度越大。
param_grid = {}
param_grid['C'] = [0.1, 5, 13, 15]
model = LogisticRegression()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator= model, param_grid = param_grid,
                    scoring = scoring, cv = kfold)
grid_result = grid.fit(X=X_train_counts_tf , y=dataset_train.target)
print('最优: {} 使用{}'.format(grid_result.best_score_,
                              grid_result.best_params_))

#=========NB算法调参
#NB有一个alpha参数，该参数是一个平滑参数，默认值为1.0。
param_grid = {}
param_grid['alpha'] = [0.001, 0.01, 0.1, 1.5]
model = MultinomialNB()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator= model, param_grid = param_grid,
                    scoring = scoring, cv = kfold)
grid_result = grid.fit(X=X_train_counts_tf , y=dataset_train.target)
print('最优: {} 使用{}'.format(grid_result.best_score_,
                              grid_result.best_params_))
'''
C=13，alpha=0.01
'''
    
#%% b）集成算法
ensembles = {} # models
ensembles['AB'] = AdaBoostClassifier()
ensembles['RF'] = RandomForestClassifier()
results = []
for key in ensembles:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(models[key], 
                   X_train_counts_tf, dataset_train.target,
                                cv=kfold, scoring= scoring)
    results.append(cv_result)
    print('{}: {} {}'.format(key, cv_result.mean(), cv_result.std()))
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(ensembles.keys())
plt.show()
'''
从执行结果看到RF的分布比较均匀，对数据的适用性比较高，更值得进一步优化。

'''    
#%% 20.9 集成算法调参！！！
#集成算法都有一个参数n_estimators，这是一个很好的可以用来调整的参数。
param_grid = {'n_estimators':[10,100,150,200]}
model = RandomForestClassifier()
kfold = KFold(n_splits=10, random_state=7)
grid = GridSearchCV(estimator= model, param_grid= param_grid,
                    scoring= scoring, cv= kfold)
grid_result = grid.fit(X=X_train_counts_tf , y=dataset_train.target)
print('最优: {} 使用{}'.format(grid_result.best_score_,
                              grid_result.best_params_))
'''150'''


#%% 20.10 确定最终模型
model = LogisticRegression(C=13)
model.fit(X=X_train_counts_tf , y=dataset_train.target)
X_test_counts_tf = tf_transformer.transform(dataset_test.data)
predictions = model.predict(X_test_counts_tf)
print(accuracy_score(dataset_test.target, predictions))
print(classification_report(dataset_test.target, predictions))

#%% 总结：
# 这类问题可以应用在垃圾邮件自动分类、新闻分类等方面。
# 在文本分类中，很重要的一点是文本特征提取，可以进一步优化，以提高准确度。
# 对中文文本分类，需要先进行分词，然后利用sklearn.dataset.base.Bunch
# 将分词后的文件加载到scikit-learn中。

#%% 草稿
#from sklearn.svm import SVC
#model = SVC(gamma=1)
#kfold = KFold(n_splits= 10, random_state= 7)
#result = cross_val_score(model, X_train_counts_tf, dataset_train.target, cv= kfold, scoring=scoring)
#print('算法评估结果(准确度acc)：mean= %.3f%%' %(result.mean()*100))
#算法评估结果(准确度acc)：mean= 90.774%
#
#


