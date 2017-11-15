# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


train_df = pd.read_csv('../data/train.csv')
sns.distplot(train_df['SalePrice'])
plt.show()