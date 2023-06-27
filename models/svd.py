import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
# import seaborn as sns
from pylab import rcParams
import string
import re
import matplotlib.pyplot as plt
import math
from matplotlib import rc
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
import matplotlib.ticker as ticker
from math import sqrt
import csv
from sklearn.metrics import mean_squared_error
import pickle

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# def data_collection():
#     """
#     Collect the Data from:
#     'recommender_data.csv' - stores number of items purchased by the user
#     'products.csv' - stores Name of the Food Item with the corresponding Food Id
#
#     Returns
#     ====
#     df : (Data Frame)
#         Stores Data from 'recommender_data.csv'
#
#     names: (Data Frame)
#         Stores Data from 'products.csv'
#
#     """
#
#     df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/recommender_data 1.csv',
#                      usecols=['user_id', 'food_id'])
#     names = pd.read_csv(
#         '/Users/mustafazaki/Downloads/Food Recommendation System/data/products.csv')
#
#     return df, names
#
#
# def data_preprocessing(df, names):
#     """
#     Performs Data Pre-processing
#
#     Params:
#     ====
#
#     df : Data Frame
#         Stores Data from 'recommender_data.csv'
#
#     names: Data Frame
#         Stores Data from 'products.csv'
#
#     Returns
#     ====
#     purchase_count : (ndarray)
#         user-item matrix with corresponding purchase count.
#
#     """
#
#     df = df.dropna()
#     df['count'] = df.groupby(['food_id', 'user_id'])[
#         'user_id'].transform('count')
#     df.drop_duplicates(keep='first', inplace=True)
#
#     df = df.groupby(['user_id', 'food_id'],
#                     as_index=False).agg({'count': 'sum'})
#     # print(df.sort_values(by='count', ascending=False))
#
#     merged_df = pd.merge(df, names, how="inner",
#                          left_on="food_id", right_on="ID")
#     merged_df.sort_values(by='count', ascending=False, inplace=True)
#     print('User-Item Data Frame:\n')
#     print(merged_df.head())
#
#     undersampled_filter = (
#         merged_df['user_id'] == '94A4CACF-0F5F-4D95-811B-266719CD2FA9') | (merged_df['user_id'] == '1')
#     undersampled_df = merged_df.loc[~undersampled_filter]
#
#     count = undersampled_df['count']
#     count_scaled = (count - count.min()) / (count.max() - count.min())
#
#     normalised_df = undersampled_df.assign(playCountScaled=count_scaled)
#
#     user_item_matrix = normalised_df.pivot(
#         index='user_id', columns='food_id', values='count')
#
#     user_item_matrix = user_item_matrix.fillna(0)
#
#     purchase_count = user_item_matrix.values
#
#     sparsity = float(len(purchase_count.nonzero()[0]))
#     sparsity /= (purchase_count.shape[0] * purchase_count.shape[1])
#     sparsity *= 100
#     print('\nSparsity: {:.2f}%'.format(100-sparsity))
#
#     return purchase_count


class SVD():
    def __init__(self, train_df, n_epochs=50, n_latent_features=3, lmbda=0.01, learning_rate=0.001, bias_lmbda=0.0001, verbose=True) -> None:
        """
        Train a matrix factorization model to predict empty
        entries in a matrix. The terminology assumes a
        ratings matrix which is ~ user x item

        Params
        ======
        purchase_count : (ndarray)
            User x Item matrix with corresponding ratings

        n_epochs : (int)
            Number of Epochs to Train the model

        n_latent_features : (int)
            Number of latent factors to use in matrix
            factorization model

        learning_rate : (float)
            Learning Rate in SGD

        lmbda : (float)
            Regularization term for user item latent factors

        bias_lmbda : (float)
            Regularization term for user item biases

        verbose : (bool)
            Whether or not to printout training progress
        """
        user_item_matrix = train_df.pivot(
            index='user_id', columns='food_id', values='count')

        self.user_item_matrix = user_item_matrix.fillna(0)

        purchase_count = self.user_item_matrix.values

        sparsity = float(len(purchase_count.nonzero()[0]))
        sparsity /= (purchase_count.shape[0] * purchase_count.shape[1])
        sparsity *= 100
        print(sparsity)

        self.purchase_count = purchase_count
        self.n_epochs = n_epochs
        self.n_latent_features = n_latent_features
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.bias_lmbda = bias_lmbda
        self.verbose = verbose

    def regularisation(self):
        """
        Helper function to calculate regularisation of user-item biases.

        Returns
        =====
        reg: (float)
            Regularisation term for user-item biases.

        """

        user_reg = 0
        item_reg = 0
        for i in range(len(self.user_bias)):
            user_reg += self.user_bias[i]**2

        user_reg = sqrt(user_reg)

        for i in range(len(self.item_bias)):
            item_reg += self.item_bias[i]**2

        item_reg = sqrt(item_reg)
        reg = self.bias_lmbda * (user_reg + item_reg)

        return reg

    def rmse(self, prediction, ground_truth):
        """
        Calculate Root Mean Square Error.

        Params
        ======
        prediction : (ndarray)
            Predicted Matrix

        ground_truth : (ndarray)
            Original Matrix

        Returns
        ======
        rmse : (float)
            Error
        """

        prediction = prediction[ground_truth.nonzero()].flatten()
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()

        rmse = sqrt(mean_squared_error(prediction, ground_truth)) + \
            self.regularisation()
        return rmse

    def train_test_split(self):
        """
        Splitting the data into Train and Test set.

        Returns
        ======
        train : (ndarray)
            Train Set
        test : (ndarray)
            Test Set
        """

        MIN_USER_PURCHASE = 35
        DELETE_COUNT = 15

        validation = np.zeros(self.purchase_count.shape)
        train = self.purchase_count.copy()

        for user in np.arange(self.purchase_count.shape[0]):
            if len(self.purchase_count[user, :].nonzero()[0]) >= MIN_USER_PURCHASE:
                val_count = np.random.choice(
                    self.purchase_count[user, :].nonzero()[0],
                    size=DELETE_COUNT,
                    replace=False
                )
                train[user, val_count] = 0
                validation[user, val_count] = self.purchase_count[user, val_count]
        return train, validation

    def predictions(self, P, Q, user_bias=0.01, item_bias=0.01):
        """
        Predict purchase count of a specific user.

        Params
        ======
        P : (ndarray)
            Latent user feature matrix
        Q : (ndarray)
            Latent item feature matrix
        user_bias : (float)
            User Bias
        item_bias : (float)
            Item Bias

        Returns
        ======
        prediction : (ndarray)
            Predicted purchase count
        """

        prediction = np.dot(P, Q) + user_bias + item_bias + self.global_bias
        return prediction

    def plot(self):
        """
        Plot loss graph
        """

        plt.plot(range(self.n_epochs), self.train_error,
                 marker='o', label='Training Data')
        plt.plot(range(self.n_epochs), self.val_error,
                 marker='v', label='Validation Data')
        plt.xlabel('Number of Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.grid()
        plt.show(block=True)

    def fit(self, X_train, X_val):
        """
        Train the model

        Params
        ======
        X_train : (ndarray)
            Train Set

        X_val : (ndarray)
            Test Set
        """

        m, n = X_train.shape
        self.user_bias = np.zeros(X_train.shape[0])
        self.item_bias = np.zeros(X_train.shape[1])
        self.global_bias = np.mean(X_train[np.where(X_train != 0)])

        self.P = 3 * np.random.rand(m, self.n_latent_features,)
        self.Q = 3 * np.random.rand(self.n_latent_features, n)

        self.train_error = []
        self.val_error = []

        users, items = X_train.nonzero()

        for epoch in range(self.n_epochs):
            for u, i in zip(users, items):
                error = X_train[u, i] - self.predictions(self.P[u, :], self.Q[:, i],
                                                         self.user_bias[u], self.item_bias[i])
                self.user_bias[u] += self.learning_rate * \
                    (error - self.bias_lmbda * self.user_bias[u])
                self.item_bias[i] += self.learning_rate * \
                    (error - self.bias_lmbda * self.item_bias[i])

                self.P[u, :] += self.learning_rate * \
                    (error * self.Q[:, i] - self.lmbda * self.P[u, :])
                self.Q[:, i] += self.learning_rate * \
                    (error * self.P[u, :] - self.lmbda * self.Q[:, i])

            train_rmse = self.rmse(self.predictions(self.P, self.Q,
                                                    self.user_bias[u], self.item_bias[i]), X_train)
            val_rmse = self.rmse(self.predictions(self.P, self.Q,
                                                  self.user_bias[u], self.item_bias[i]), X_val)
            self.train_error.append(train_rmse)
            self.val_error.append(val_rmse)
            print(f'---------------- {epoch + 1} Done ----------------')

        if self.verbose:
            self.plot()

        return self

    def predict(self, X_train, user_index):
        """
        Predict purchase count of all the food items which a user has never purchased.

        Params
        =====
        X_train : (ndarray)
            user-item matrix
        user_index: (int)
            Index of he user

        Returns
        =====
        y_pred : (1darray)
            List of predicted purchase count of all the items user has not purchased.
        """

        y_hat = self.predictions(self.P, self.Q)
        predictions_index = np.where(X_train[user_index, :] == 0)[0]
        y_pred = y_hat[user_index, predictions_index].flatten()
        return y_pred


    # def save(self, filename='/Users/mustafazaki/Downloads/Food Recommendation System/models/saved models/svd_model.pkl'):
    #     # Save the model
    #     with open(filename, 'wb') as f:
    #         pickle.dump(self, f)
    #     print('Model was successfully saved')
    #
    # @staticmethod
    # def load(filename):
    #     # Check if the file exists
    #     if not os.path.isfile(filename):
    #         raise Exception(f'File {filename} does not exist')
    #     # Load the model
    #     with open(filename, 'rb') as f:
    #         model = pickle.load(f)
    #     return model


    def create_recommendations(self, user_id, X, n=10):
        print(f'\n------------------- SVD: {user_id} ---------------------\n')

        user_index = self.user_item_matrix.index.get_loc(user_id)

        prediction_indices = np.where(X[user_index, :] == 0)[0]
        recommendations = self.predict(X, user_index)

        purchase_indices = np.where(X[user_index, :] > 0)[0]
        existing_purchase = X[user_index, purchase_indices]

        ### Existing Purchase History

        food_ids = self.user_item_matrix.columns[purchase_indices]
        food_purchase = pd.DataFrame(data=dict(foodId=food_ids, purchase_count=existing_purchase))
        top_food_items = food_purchase.sort_values("purchase_count", ascending=False).head(n)

        products_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')

        current_purchase = products_df[products_df.ID.isin(top_food_items.foodId)].reset_index(drop=True)
        current_purchase['purchase_count'] = pd.Series(top_food_items.purchase_count.values)
        print('Items purchased by the user:')
        print(current_purchase)

        ### Predicted Purchase History

        food_ids = self.user_item_matrix.columns[prediction_indices]
        food_purchase = pd.DataFrame(data=dict(foodId=food_ids, purchase_count=recommendations))
        top_food_items = food_purchase.sort_values("purchase_count", ascending=False).head(n)

        # products_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')

        food_recommendations = products_df[products_df.ID.isin(top_food_items.foodId)].reset_index(drop=True)
        food_recommendations['purchase_count'] = pd.Series(top_food_items.purchase_count.values)

        print('Items recommended to the user:')
        print(food_recommendations)

        return food_recommendations.sort_values("purchase_count", ascending=False)




# def main():
#     df, names = data_collection()
#
#     X = data_preprocessing(df, names)
#     svd_model = Recommender(X)
#     X_train, X_test = svd_model.train_test_split()
#     svd_model.fit(X_train, X_test)


# if __name__ == '__main__':
#     main()
