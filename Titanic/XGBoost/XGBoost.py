# coding: utf-8


import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np


train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')


# 填充空值，用中位数填充数值型空值，用众数填充字符型空值
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.fill)


train_df['Family'] = train_df['Parch'] + train_df['SibSp']
test_df['Family'] = test_df['Parch'] + test_df['SibSp']
# print(train_df.loc[:,['Family','Parch','SibSp']])

feature_columns_to_use = ['Pclass', 'Age', 'Sex', 'Fare', 'Family', 'Embarked']
nonnumeric_columns = ['Sex', 'Embarked']


big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_Imputed = DataFrameImputer().fit_transform(big_X)


le = LabelEncoder()
for feature in nonnumeric_columns:
    big_X_Imputed[feature] = le.fit_transform(big_X_Imputed[feature])


X_train = big_X_Imputed[0:train_df.shape[0]].as_matrix()
Y_train = train_df['Survived']
X_test = big_X_Imputed[train_df.shape[0]:].as_matrix()


gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)
gbm.fit(X_train, Y_train)
Y_pred = gbm.predict(X_test)
print(gbm.score(X_train, Y_train))


submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    "Survived": Y_pred
})
# print(submission.head())
submission.to_csv('../submission/submission_7.csv', index=False)