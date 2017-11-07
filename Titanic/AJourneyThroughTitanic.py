# coding: utf-8


# import 相关库
import pandas as pd
from pandas import Series, DataFrame

import numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.svm import SVC, LinearSVC              # 支持向量机
from sklearn.ensemble import RandomForestClassifier  # 随即森林
from sklearn.neighbors import KNeighborsClassifier  # K 近邻
from sklearn.naive_bayes import GaussianNB          # 朴素贝叶斯


# 获取训练集和测试集
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')
# 预览数据集
# print(train_df.head())
# print(train_df.info())
# print(test_df.info())


# 抛弃不需要的列，这些列对分析和预测无用
train_df = train_df.drop(['Name', 'PassengerId', 'Ticket'], axis=1)
test_df = test_df.drop(['Name', 'Ticket'], axis=1)


# Embarked，补充 train_df 中的缺失值，有两条数据缺少 Embarked 信息，用最常见的值 S 填充
train_df['Embarked'] = train_df['Embarked'].fillna('S')


# 画图
sns.factorplot('Embarked', 'Survived', data=train_df, size=4, aspect=3)
plt.savefig('Embarked_1.png')
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize=(15,5))
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=train_df, order=[1,0], ax=axis2)
# 以 embarked 分组，并计算每组中幸存乘客的平均值
embark_perc = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)
plt.savefig('Embarked_2.png')