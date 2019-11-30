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


#%% 
一个很好的实践机器学习项目的方法，
是使用从UCI机器学习仓库获取的数据集开启一个机器学习项目。
#%% 机器学习项目的python模板
通常是六个步骤：
1.定义问题
a）导入类库
b）导入数据集

2.理解数据
a）描述性统计
b）数据可视化

3.数据准备
a）数据清洗
b）特征选择
c）数据转换

4.评估算法
a）分离数据集
b）定义模型评估标准
c）算法审查
d）算法比较

5.优化模型
a）算法调参
b）集成算法

6.结果部署
a）预测评估数据集
b）利用整个数据集生成模型
c）序列化模型

#%% 使用模板的小技巧
（1）快速执行一遍：
（2）整个流程不是线性的，需要多次重复步骤3和4
（3）尝试每一个步骤，即使你认为不适用，也不要跳过，而是减少该步骤所做的贡献。

#%% 草稿