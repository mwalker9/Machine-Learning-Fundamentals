## 1. Recap ##

import pandas as pd
import numpy as np
np.random.seed(1)

dc_listings = pd.read_csv('dc_airbnb.csv')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings.info()

## 2. Removing features ##

problematic_columns = ['room_type', 'city', 'state', 'latitude', 'longitude', 'zipcode', 'host_response_rate', 'host_acceptance_rate', 'host_listings_count']
dc_listings.drop(columns=problematic_columns, inplace=True)

## 3. Handling missing values ##

dc_listings.drop(columns=['cleaning_fee', 'security_deposit'], inplace=True)
dc_listings.dropna(axis=0, inplace=True)
dc_listings.info()

## 4. Normalize columns ##

normalized_listings = (dc_listings - dc_listings.mean())/(dc_listings.std())
normalized_listings['price'] = dc_listings['price']
print(normalized_listings.head(3))

## 5. Euclidean distance for multivariate case ##

from scipy.spatial import distance
first_listing = normalized_listings.iloc[0][["accommodates", "bathrooms"]]
second_listing = normalized_listings.iloc[4][["accommodates", "bathrooms"]]
first_fifth_distance = distance.euclidean(first_listing, second_listing)
print(first_fifth_distance)

## 7. Fitting a model and making predictions ##

from sklearn.neighbors import KNeighborsRegressor

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')
knn.fit(train_df[['accommodates', 'bathrooms']], train_df['price'])
predictions = knn.predict(test_df[['accommodates', 'bathrooms']])


## 8. Calculating MSE using Scikit-Learn ##

from sklearn.metrics import mean_squared_error

train_columns = ['accommodates', 'bathrooms']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
knn.fit(train_df[train_columns], train_df['price'])
predictions = knn.predict(test_df[train_columns])
two_features_mse = mean_squared_error(test_df['price'], predictions)
two_features_rmse = np.sqrt(two_features_mse)
print(two_features_mse, two_features_rmse)

## 9. Using more features ##

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')
knn.fit(train_df[features], train_df['price'])
four_predictions = knn.predict(test_df[features])
four_mse = mean_squared_error(test_df['price'], four_predictions)
four_rmse = np.sqrt(four_mse)
print(four_mse, four_rmse)


## 10. Using all features ##

all_columns_but_price = [x for x in train_df.columns.values if x != 'price']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')
knn.fit(train_df[all_columns_but_price], train_df['price'])
all_features_predictions = knn.predict(test_df[all_columns_but_price])
all_features_mse = mean_squared_error(test_df['price'], all_features_predictions)
all_features_rmse = np.sqrt(all_features_mse)
print(all_features_mse, all_features_rmse)
