# coding: utf-8


# import 相关库
import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.svm import SVC, LinearSVC              # 支持向量机
from sklearn.ensemble import RandomForestClassifier # 随即森林
from sklearn.neighbors import KNeighborsClassifier  # K 近邻
from sklearn.naive_bayes import GaussianNB          # 朴素贝叶斯
from sklearn.tree import DecisionTreeClassifier     # 决策树


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
plt.close('all')
# 要么在预测时考虑 Embarked 列，删除 S 虚拟变量，保留 C 和 Q，因为具有较好的生存概率
# 要么不为 Embarked 列创建虚拟变量，抛弃 Embarked 元素，因为从逻辑上讲 Embarked 对预测无用
embark_dummies_titanic = pd.get_dummies(train_df['Embarked'])
embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
embark_dummies_test = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['S'], axis=1, inplace=True)
train_df = train_df.join(embark_dummies_titanic)
test_df = test_df.join(embark_dummies_test)
train_df.drop(['Embarked'], axis=1, inplace=True)
test_df.drop(['Embarked'], axis=1, inplace=True)


# Fare，补充 test_df 中的缺失值，有一条数据缺少 Fare 信息，使用中位数填充
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)
# 从 float 型转换为 int 型
train_df['Fare'] = train_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)
# 分别获取幸存者和遇难者的 Fare
fare_not_survived = train_df['Fare'][train_df['Survived'] == 0]
fare_survived = train_df['Fare'][train_df['Survived'] == 1]
# 计算平均值和标准差
average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])
# 画图
train_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100, xlim=(0,50))
plt.savefig('Fare_1.png')
average_fare.index.names = std_fare.index.names = ['Survived']
average_fare.plot(yerr=std_fare, kind='bar', legend=False)
plt.savefig('Fare_2.png')
plt.close('all')


# 年龄
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15,4))
axis1.set_title('Original Age Values - Titanic')
axis2.set_title('New Age Values - Titanic')
# 获取训练集中 Age 的平均值、标准差和 null 的个数
average_age_train = train_df['Age'].mean()
std_age_train = train_df['Age'].std()
count_nan_age_train = train_df['Age'].isnull().sum()
# 获取测试集中 Age 的平均值、标准差和 null 的个数
average_age_test = test_df['Age'].mean()
std_age_test = test_df['Age'].std()
count_nan_age_test = test_df['Age'].isnull().sum()
# 在（mean-std, mean+std）之间生成随机数
rand_1 = np.random.randint(average_age_train-std_age_train, average_age_train+std_age_train, size=count_nan_age_train)
rand_2 = np.random.randint(average_age_test-std_age_test, average_age_test+std_age_test, size=count_nan_age_test)
# 绘制原始年龄，抛弃空值并转化为 int 型
train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
# 使用生成的随机数填充空值
train_df['Age'][np.isnan(train_df['Age'])] = rand_1
test_df['Age'][np.isnan(test_df['Age'])] = rand_2
# 从 float 型转换为 int 型
train_df['Age'] = train_df['Age'].astype(int)
test_df['Age'] = test_df['Age'].astype(int)
# 绘制新年龄值
train_df['Age'].hist(bins=70, ax=axis2)
plt.savefig('Age_1.png')
plt.close('all')
# 绘制幸存和遇难与年龄的关系，观察其峰值
facet = sns.FacetGrid(train_df, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plt.savefig('Age_2.png')
# 按照年龄计算存活率
fig, axis1 = plt.subplots(1, 1, figsize=(18,4))
average_age = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
plt.savefig('Age_3.png')


# Cabin，有太多缺失值，对预测无用
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)


# Family，计算家庭成员，观察是否和幸存相关
train_df['Family'] = train_df['Parch'] + train_df['SibSp']
train_df['Family'].loc[train_df['Family'] > 0] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0
test_df['Family'] = test_df['Parch'] + test_df['SibSp']
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0
# 移除 Parch 和 SibSp 列
train_df.drop(['SibSp', 'Parch'], axis=1)
test_df.drop(['SibSp', 'Parch'], axis=1)
# 画图
fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10,5))
sns.countplot(x='Family', data=train_df, order=[1,0], ax=axis1)
family_perc = train_df[['Family', 'Survived']].groupby('Family', as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)
axis1.set_xticklabels(['With Family', 'Alone'], rotation=0)
plt.savefig('Family.png')
plt.close('all')


# Sex，通过观察年龄曲线可以知道，Age < 15 时存活远大于遇难，因此，将乘客分为男性、女性、儿童
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex
train_df['Person'] = train_df[['Age', 'Sex']].apply(get_person, axis=1)
test_df['Person'] = test_df[['Age', 'Sex']].apply(get_person, axis=1)
# 用 Person 列替换 Sex 列
train_df.drop(['Sex'], axis=1, inplace=True)
test_df.drop(['Sex'], axis=1, inplace=True)
# 为 Person 列创建虚拟变量，舍弃 male，因为其平均存活率最低
person_dummies_train = pd.get_dummies(train_df['Person'])
person_dummies_train.columns = ['Child', 'Male', 'Female']
person_dummies_train.drop(['Male'], axis=1, inplace=True)
person_dummies_test = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child', 'Male', 'Female']
person_dummies_test.drop(['Male'], axis=1, inplace=True)
train_df.join(person_dummies_train)
test_df.join(person_dummies_test)
# 画图
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10,5))
sns.countplot(x='Person', data=train_df, ax=axis1)
person_prec = train_df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_prec, ax=axis2, order=['male','female','child'])
train_df.drop(['Person'], axis=1, inplace=True)
test_df.drop(['Person'], axis=1, inplace=True)
plt.savefig('Sex.png')
plt.close('all')


# Pclass
sns.factorplot('Pclass', 'Survived', order=[1,2,3], data=train_df, size=5)
plt.savefig('Pclass_1.png')
plt.close('all')
# 为 Pclass 创建虚拟变量，并删除存活率最低的 3
pclass_dummies_train = pd.get_dummies(train_df['Pclass'])
pclass_dummies_train.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_train.drop(['Class_3'], axis=1, inplace=True)
pclass_dummies_test = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1', 'Class_2', 'Class_3']
pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)
train_df.drop(['Pclass'], axis=1, inplace=True)
test_df.drop(['Pclass'], axis=1, inplace=True)
train_df = train_df.join(pclass_dummies_train)
test_df = test_df.join(pclass_dummies_test)


# 定义训练集和测试集
X_train = train_df.drop(['Survived'], axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop(['PassengerId'], axis=1).copy()


# 逻辑回归，在训练集准确率 0.717171717172
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
# print(logreg.score(X_train, Y_train))


# 随机森林，在训练集准确率 0.943883277217
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
# print(random_forest.score(X_train, Y_train))


# 决策树
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
print(decision_tree.score(X_train, Y_train))


# 生成提交结果
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('./data/submission_4.csv', index=False)