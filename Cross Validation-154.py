## 1. Introduction ##

import numpy as np
import pandas as pd

dc_listings = pd.read_csv("dc_airbnb.csv")
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
randomized_rows = np.random.permutation(dc_listings.shape[0])
dc_listings = dc_listings.iloc[randomized_rows]
split_one = dc_listings.iloc[:1862]
split_two = dc_listings.iloc[1862:]

## 2. Holdout Validation ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

train_one = split_one
test_one = split_two
train_two = split_two
test_two = split_one

knn = KNeighborsRegressor()
knn.fit(train_one[['accommodates']], train_one['price'])
predictions = knn.predict(test_one[['accommodates']])
iteration_one_rmse = np.sqrt(mean_squared_error(test_one['price'], predictions))

knn.fit(train_two[['accommodates']], train_two['price'])
predictions = knn.predict(test_two[['accommodates']])
iteration_two_rmse = np.sqrt(mean_squared_error(test_two['price'], predictions))
avg_rmse = np.mean([iteration_one_rmse, iteration_two_rmse])

## 3. K-Fold Cross Validation ##

dc_listings['fold'] = np.nan
dc_listings.loc[0:745, 'fold'] = 1
dc_listings.loc[745:1490, 'fold'] = 2
dc_listings.loc[1490:2234, 'fold'] = 3
dc_listings.loc[2234:2978, 'fold'] = 4
dc_listings.loc[2978:3723, 'fold'] = 5
dc_listings['fold'].value_counts()

## 4. First iteration ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor()
folds = dc_listings.loc[dc_listings['fold'] != 1]
knn.fit(folds[['accommodates']], folds['price'])
labels = knn.predict(dc_listings.loc[dc_listings['fold'] == 1][['accommodates']])
iteration_one_rmse = np.sqrt(mean_squared_error(labels, dc_listings.loc[dc_listings['fold'] == 1]['price']))


## 5. Function for training models ##

# Use np.mean to calculate the mean.
import numpy as np
fold_ids = [1,2,3,4,5]

def train_and_validate(df, folds):
    rmse = []
    for n in folds:
        knn = KNeighborsRegressor()
        train_set = df.loc[df['fold'] != n]
        test_set = df.loc[df['fold'] == n]
        knn.fit(train_set[['accommodates']], train_set['price'])
        predictions = knn.predict(test_set[['accommodates']])
        rmse.append(np.sqrt(mean_squared_error(predictions, test_set['price'])))
    return rmse

rmses = train_and_validate(dc_listings, fold_ids)
avg_rmse = np.mean(rmses)
print(rmses, avg_rmse)


## 6. Performing K-Fold Cross Validation Using Scikit-Learn ##

from sklearn.model_selection import cross_val_score, KFold

kf = KFold(5, shuffle=True, random_state=1)
knn = KNeighborsRegressor()
mses = cross_val_score(knn, dc_listings[['accommodates']], dc_listings['price'], scoring='neg_mean_squared_error', cv=kf)
avg_rmse = np.mean(np.sqrt(np.absolute(mses)))


## 7. Exploring Different K Values ##

from sklearn.model_selection import cross_val_score, KFold

num_folds = [3, 5, 7, 9, 10, 11, 13, 15, 17, 19, 21, 23]

for fold in num_folds:
    kf = KFold(fold, shuffle=True, random_state=1)
    model = KNeighborsRegressor()
    mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
    rmses = np.sqrt(np.absolute(mses))
    avg_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)
    print(str(fold), "folds: ", "avg RMSE: ", str(avg_rmse), "std RMSE: ", str(std_rmse))