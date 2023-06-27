import os, sys
sys.path.append(os.path.abspath('/Users/mustafazaki/Downloads/Food Recommendation System/preprocessing'))

import pickle
import numpy as np
import pandas as pd
import csv
import torch
from sklearn.preprocessing import LabelEncoder
from preprocessing import BinaryNormalizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Device: ', device)


class EASE:
    def __init__(self, train_df, regularization=0.05, user_col='user_id', item_col='food_id', value_col='count'):
        """
        Initialize EASE model.

        Parameters:
        train_df (pandas.DataFrame): DataFrame with user-item interactions.
        regularization (float): Regularization parameter. Default is 0.05.
        user_col (str): Column name of user IDs in the DataFrame. Default is 'user_id'.
        item_col (str): Column name of item IDs in the DataFrame. Default is 'food_id'.
        item_col (str): Column name of values in the DataFrame. Default is 'count'.

        Returns:
        None
        """

        self.df = train_df
        self.regularization = regularization
        self.item_col = item_col
        self.user_col = user_col

        # Create label encoders for user and item IDs
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        # Store the names of user and item ID columns
        # self.user_col = user_col
        # self.item_col = item_col

        # Create a new column for Item IDs
        self.df['encoded_item'] = self.item_encoder.fit_transform(
            self.df[self.item_col])

        # Create a new dataframe with encoded item IDs as the index
        self.encoded_items_df = self.df[[self.item_col, 'encoded_item']]
        self.encoded_items_df = self.encoded_items_df.set_index(self.item_col)

    def fit(self):
        user_indices = self.user_encoder.fit_transform(self.df[self.user_col])
        item_indices = self.item_encoder.transform(self.df[self.item_col])
        purchase_count = self.df['count'] / max(self.df['count'])

        users_tensor = torch.LongTensor(user_indices)
        items_tensor = torch.LongTensor(item_indices)
        values_tensor = torch.FloatTensor(purchase_count)

        indices = torch.LongTensor(
            np.array([users_tensor.numpy().astype(np.int64), items_tensor.numpy().astype(np.int64)]))

        # Constructing sparse tensor X with the user-item matrix
        self.X = (torch.sparse.FloatTensor(indices, values_tensor)).to(device)

        # Building item-item matrix
        G = (self.X.to_dense().t() @ self.X.to_dense()).to(device)
        G += (torch.eye(G.shape[0]) * self.regularization).to(device)
        P = torch.linalg.inv(G).to(device)
        B = (P / (-1 * torch.diag(P))).to(device)
        self.B = B.fill_diagonal_(fill_value=0, wrap=False)
        self.B = self.B + (torch.eye(B.shape[0])).to(device)

    def predict(self, prediction_df, k=10, remove_prev=True):
        """
        Predict top k items for each user in the input DataFrame.

        Parameters:
        pred_df (pandas.DataFrame): DataFrame with user-item interactions for which recommendations are to be made.
        k (int): Number of recommendations to be made for each user. Default is 10.
        remove_prev (bool): Decides if the previous items purchased by the user should be removed from the predictions. Default is True.

        Returns:
        recommendations_df (pandas.DataFrame): DataFrame with columns 'user_id' and 'recommendations'. 'recommendations' contains a list of k recommended item IDs for each user.
        """

        # Encoding items of integer IDs by merging 'self.encoded_items_df'
        merged_df = prediction_df.merge(
            self.encoded_items_df, left_on=self.item_col, right_index=True).drop_duplicates()

        # Grouping by user
        grouped = merged_df.groupby(self.user_col)
        preds = []
        for group_name, group_df in grouped:
            # Building a user vector with purchase counts

            user_vector = torch.zeros(self.B.shape[1]).to(device)
            for item, count in zip(group_df['encoded_item'], group_df['count']):
                user_vector[item] = count / max(group_df['count'])

            # Multiplying user vector by item-item matrix
            pred_vector = user_vector @ self.B

            if remove_prev:
                pred_vector -= user_vector

            # Getting top-k items
            top_items = torch.topk(pred_vector, k, sorted=True).indices

            # Decoding item IDs to original format
            item_ids = self.item_encoder.inverse_transform(
                top_items.cpu().numpy())

            preds.append((group_name, item_ids))

        # Creating recommendations DataFrame
        recommendations_df = pd.DataFrame(
            preds, columns=[self.user_col, 'recommendations'])

        return recommendations_df

    def save(self, filename='/Users/mustafazaki/Downloads/Food Recommendation System/models/saved models/ease_model.pkl'):
        # Save the model
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print('Model was successfully saved')

    @staticmethod
    def load(filename='/Users/mustafazaki/Downloads/Food Recommendation System/models/saved models/ease_model.pkl'):
        # Check if the file exists
        if not os.path.isfile(filename):
            raise Exception(f'File {filename} does not exist')
        # Load the model
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model


# def main():
#     binary_ease = BinaryNormalizer()
#     preprocessed_df = binary_ease.fit_transform()
#
#
#     # processed_df = data_preprocessing(df, names)
#     ease_model = EASE(preprocessed_df)
#     ease_model.fit()
#     test_df = pd.DataFrame({
#         'user_id': ['gib', 'gib', 'gib', 'kld', 'kld', 'kld', 'sds', 'sds', 'sds'],
#         'food_id': ['u09132', 'u09050', 'u09316', 'u09003', 'u01117', 'u09050', 'g15547', 'b45252410', 'g16050'],
#         'count': [12, 3, 4, 10, 5, 6, 4, 5, 8]})
#     print('\nTest Dataframe: \n', test_df)
#
#     recommendations = ease_model.predict(test_df, k=10)
#     print('\n---------------------------------------------')
#     print('\nRecommendations Dataframe: \n', recommendations)
#
#     print('\n---------------------------------------------')
#     # print('\nNames Dataframe: \n', names)
#     names.set_index('ID', inplace=True)
#     recommendations.set_index('user_id', inplace=True)
#     # print(recommendations[0])
#     # user, group_df = test_df.groupby('user_id')
#
#     for user in recommendations.index:
#         print('\n---------------------------------------------')
#         print(f'\n\nItems Purchased by the user: {user}')
#         for purchased_item in test_df[test_df['user_id'] == user]['food_id']:
#             print(
#                 f"{purchased_item} : {names.loc[purchased_item]['Product Name']}")
#         print(f"\nItems Recommended to the user: {user}")
#         for recommended_item in recommendations.loc[user]['recommendations']:
#             print(
#                 f"{recommended_item} : {names.loc[recommended_item]['Product Name']}")
#
#
#
# if __name__ == '__main__':
#     main()
