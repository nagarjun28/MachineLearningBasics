import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import sklearn

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

sns.lmplot('RM', 'Price', data=boston_df)

X = np.vstack(boston_df.RM)

Y = boston_df.Price

# to write in form of a matrix to form Y=mX+c -> [X 1]

X = np.array([[value, 1] for value in X], dtype=float)
m, b = np.linalg.lstsq(X, Y, rcond=None)[0]
plt.plot(boston_df.RM, boston_df.Price, 'o')
x = boston_df.RM
plt.plot(x, m * x + b, 'r', label='best fit line')

result = np.linalg.lstsq(X, Y, rcond=None)
error_total = result[1]
rmse = np.sqrt(error_total / len(X))

print('The root mean square error is ', rmse)
lreg = LinearRegression()

X_multi = boston_df.drop('Price', 1)
Y_target = boston_df.Price
lreg.fit(X_multi, Y_target)

coeff_df = DataFrame(boston_df.columns)
coeff_df.columns = ['Features']
coeff_df['Coefficient Estimate'] = Series(lreg.coef_)

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, boston_df.Price)
lreg = LinearRegression()
lreg.fit(X_train, Y_train)

pred_train = lreg.predict(X_train)
pred_test = lreg.predict(X_test)

print("Fit a model X_train and calculate the MSE with Y_train: ", np.mean((Y_train - pred_train) ** 2))
print("Fit the model X_train and calculate MSE with X_test and Y_test ", np.mean((Y_test - pred_test) ** 2))

train = plt.scatter(pred_train, (pred_train - Y_train), c='b', alpha=0.5)
test = plt.scatter(pred_test,(pred_test-Y_test),c='r',alpha=0.5)
plt.hlines(y=0,xmin=-10,xmax=50)
plt.legend((train,test),('Training','Test'),loc='lower left')
plt.title('residual plots')