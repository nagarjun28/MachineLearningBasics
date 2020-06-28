import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston

boston = load_boston()
# plt.hist(boston.target,bins=50)
# plt.xlabel('Prices in 1000$')
# plt.ylabel('Number of houses')

# plt.scatter(boston.data[:,5],boston.target)
# plt.ylabel('Price in 1000$')
# plt.xlabel('Number of Rooms')

# convert into a data frame to use seaborn

boston_df = DataFrame(boston.data)
boston_df.columns = boston.feature_names

boston_df['Price'] = boston.target

sns.lmplot('RM','Price',data=boston_df)