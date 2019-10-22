## 1. Recap ##

import pandas as pd
train_df = pd.read_csv('dc_airbnb_train.csv')
test_df = pd.read_csv('dc_airbnb_test.csv')

## 2. Hyperparameter optimization ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
mse_values = []
cols = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
for k in range(1, 6):
    knn = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    knn.fit(train_df[cols], train_df['price'])
    predictions = knn.predict(test_df[cols])
    mse_values.append(mean_squared_error(test_df['price'], predictions))
print(mse_values)
         
    

## 3. Expanding grid search ##

mse_values = []
cols = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    knn.fit(train_df[cols], train_df['price'])
    predictions = knn.predict(test_df[cols])
    mse_values.append(mean_squared_error(test_df['price'], predictions))
print(mse_values)

## 4. Visualizing hyperparameter values ##

import matplotlib.pyplot as plt

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
hyper_params = [x for x in range(1, 21)]
mse_values = list()

for hp in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=hp, algorithm='brute')
    knn.fit(train_df[features], train_df['price'])
    predictions = knn.predict(test_df[features])
    mse = mean_squared_error(test_df['price'], predictions)
    mse_values.append(mse)
plt.scatter(hyper_params, mse_values)
plt.show()

## 5. Varying Hyperparameters ##

hyper_params = [x for x in range(1,21)]
mse_values = list()
cols = [c for c in train_df.columns.values if c != 'price']
for k in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    knn.fit(train_df[cols], train_df['price'])
    predictions = knn.predict(test_df[cols])
    mse_values.append(mean_squared_error(test_df['price'], predictions))

plt.scatter(hyper_params, mse_values)
plt.show()

## 6. Practice the workflow ##

import numpy as np
two_features = ['accommodates', 'bathrooms']
three_features = ['accommodates', 'bathrooms', 'bedrooms']
hyper_params = [x for x in range(1,21)]
# Append the first model's MSE values to this list.
two_mse_values = list()
for k in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    knn.fit(train_df[two_features], train_df['price'])
    predictions = knn.predict(test_df[two_features])
    two_mse_values.append(mean_squared_error(test_df['price'], predictions)) 

# Append the second model's MSE values to this list.
three_mse_values = list()
    
for k in hyper_params:
    knn = KNeighborsRegressor(n_neighbors=k, algorithm='brute')
    knn.fit(train_df[three_features], train_df['price'])
    predictions = knn.predict(test_df[three_features])
    three_mse_values.append(mean_squared_error(test_df['price'], predictions))

two_hyp_mse = dict()
three_hyp_mse = dict()
min_pos = np.argmin(two_mse_values)
two_hyp_mse[min_pos+1] = two_mse_values[min_pos]
min_pos = np.argmin(three_mse_values)
three_hyp_mse[min_pos+1] = three_mse_values[min_pos]