## 1. Testing quality of predictions ##

import pandas as pd
import numpy as np
dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
train_df = dc_listings.iloc[0:2792]
test_df = dc_listings.iloc[2792:]

def predict_price(new_listing):
    ## DataFrame.copy() performs a deep copy
    temp_df = train_df.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbor_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbor_prices.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df.accommodates.apply(predict_price)

## 2. Error Metrics ##

import numpy as np
mae = np.mean(np.absolute(test_df.predicted_price - test_df.price))

## 3. Mean Squared Error ##

raw_error = (test_df.predicted_price - test_df.price)
mse = 1/test_df.shape[0] * np.dot(raw_error, raw_error)

## 4. Training another model ##

train_df = dc_listings.iloc[0:2792]
test_df = dc_listings.iloc[2792:]

def predict_price(new_listing):
    temp_df = train_df.copy()
    temp_df['distance'] = temp_df['bathrooms'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbors_prices.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df.bathrooms.apply(predict_price)
raw_error = test_df.predicted_price - test_df.price
squared_error = np.dot(raw_error, raw_error)
mse = 1/test_df.shape[0] * squared_error
print(mse)

## 5. Root Mean Squared Error ##

def predict_price(new_listing):
    temp_df = train_df.copy()
    temp_df['distance'] = temp_df['bathrooms'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors_prices = temp_df.iloc[0:5]['price']
    predicted_price = nearest_neighbors_prices.mean()
    return(predicted_price)

test_df['predicted_price'] = test_df['bathrooms'].apply(lambda x: predict_price(x))
test_df['squared_error'] = (test_df['predicted_price'] - test_df['price'])**(2)
mse = test_df['squared_error'].mean()
rmse = np.sqrt(mse)

## 6. Comparing MAE and RMSE ##

errors_one = pd.Series([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10])
errors_two = pd.Series([5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 10, 5, 1000])
mae_one = np.sum(np.absolute(errors_one))/errors_one.shape[0]
rmse_one = np.sqrt(np.dot(errors_one, errors_one)/errors_one.shape[0])
mae_two = np.sum(np.absolute(errors_two))/errors_two.shape[0]
rmse_two = np.sqrt(np.dot(errors_two, errors_two)/errors_two.shape[0])