## 2. Introduction to the data ##

import pandas as pd

dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.head(1))

## 4. Euclidean distance ##

import numpy as np
first_distance = abs(3 - dc_listings.loc[0, 'accommodates'])
print(first_distance)

## 5. Calculate distance for all observations ##

def distance(item):
    return abs(item - 3)
dc_listings['distance'] = dc_listings.accommodates.apply(distance)
print(dc_listings.distance.value_counts())

## 6. Randomizing, and sorting ##

import numpy as np
np.random.seed(1)
dc_listings = dc_listings.loc[np.random.permutation(dc_listings.index.values)]
dc_listings = dc_listings.sort_values(by='distance')
print(dc_listings.price)

## 7. Average price ##

stripped_commas = dc_listings.price.str.replace(',', '').str.replace('$', '')
dc_listings.price = pd.to_numeric(stripped_commas)
mean_price = np.mean(dc_listings.head(5).price)
print(mean_price)

## 8. Function to make predictions ##

# Brought along the changes we made to the `dc_listings` Dataframe.
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

def get_distance(neighbor, num_accommodates):
    return abs(num_accommodates - neighbor)

def predict_price(new_listing):
    temp_df = dc_listings.copy()
    temp_df['distance'] = temp_df.accommodates.apply(get_distance, args=(new_listing,))
    knn_price = temp_df.sort_values(by='distance').head(5).price
    return np.mean(knn_price)

acc_one = predict_price(1)
acc_two = predict_price(2)
acc_four = predict_price(4)