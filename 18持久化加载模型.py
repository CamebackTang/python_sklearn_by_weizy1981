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

#%% 持久化加载模型
 在实际的项目中，需要将生成的模型序列化，并将其发布到生产环境。
 当新数据出现时，需要反序列化已保存的模型，然后用其预测新的数据。
 （1）模型序列化和重用的重要性
 （2）如何通过pickle来序列化和反序列化机器学习模型。
 （3）如何通过joblib来序列化和反序列化机器学习模型。
当模型训练需要花费大量的时间时，模型序列化就尤为重要。

#%% （2）如何通过pickle来序列化和反序列化机器学习模型
#pickle是标准的python序列化的方法，可以通过它序列化模型，并将其保存到文件中。
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import dump
from pickle import load
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size= 0.33, random_state= 4)
#训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

#保存模型
model_file = 'finalized_model.sav'
with open(model_file, 'wb') as model_f:
    dump(model, model_f) # 模型序列化

#加载模型
with open(model_file, 'rb') as model_f:
    loaded_model = load(model_f) # 模型反序列化
    result = loaded_model.score(X_test, y_test)
    print('算法评估结果：%.3f%%' %(result*100))
    
#%%（3）如何通过joblib来序列化和反序列化机器学习模型。
#joblib是scipy生态环境的一部分，提供了通用的工具来序列化和反序列化python的对象 。   
#通过joblib序列化对象时会采用Numpy的格式保存数据，
#这对某些保存数据到模型中的算法非常有效，如k近邻。
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.externals.joblib import dump
from sklearn.externals.joblib import load
model = LogisticRegression()
model.fit(X_train, y_train)

#保存模型
model_file = 'finalized_model_joblib.sav'
with open(model_file, 'wb') as model_f:
    dump(model, model_f) # 模型序列化

#加载模型
with open(model_file, 'rb') as model_f:
    loaded_model = load(model_f) # 模型反序列化
    result = loaded_model.score(X_test, y_test)
    print('算法评估结果：%.3f%%' %(result*100))

#%% 18.3 生成模型的技巧
在生成机器学习模型时，需要考虑以下几个问题：
（1）python的版本：要记录下python的版本，大部分情况下，
在序列化和反序列化模型时，需要使用相同的python版本。
（2）类库版本：同样需要记录所有的主要类库的版本，
不仅需要scipy和scikit-learn版本一致，其他类库版本也需要一致。
（3）手动序列化：有时需要手动序列化算法参数，
这样可以直接在scikit-learn中或其他平台重现这个模型。
我们通常会花费大量的时间在选择算法和参数调整上，
将这个过程手动记录下来比序列化模型更有价值

#%% 草稿