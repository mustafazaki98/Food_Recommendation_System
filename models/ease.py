import os, sys
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

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
        value_col (str): Column name of values in the DataFrame. Default is 'count'.

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

        # Create a new column for Item IDs
        self.df['encoded_item'] = self.item_encoder.fit_transform(
            self.df[self.item_col])

        # Create a new dataframe with encoded item IDs as the index
        self.encoded_items_df = self.df[[self.item_col, 'encoded_item']]
        self.encoded_items_df = self.encoded_items_df.set_index(self.item_col)


    def fit(self):
        user_indices = self.user_encoder.fit_transform(self.df[self.user_col])
        item_indices = self.item_encoder.transform(self.df[self.item_col])
        purchase_count = self.df['count']

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

    def predict(self, user_id, k=10):
        """
        Predict top k items for the given user ID.

        Parameters:
        user_id (str): User ID for which recommendations are to be made.
        k (int): Number of recommendations to be made. Default is 10.

        Returns:
        recommendations (list): List of k recommended item IDs for the user.
        """

        # Find the user's index from the label encoder
        user_index = self.user_encoder.transform([user_id])

        # Check if the user exists in the training data
        if len(user_index) == 0:
            return []

        user_index = user_index[0]

        # Get the user's interaction vector from the user-item matrix
        user_vector = self.X[user_index].to_dense()

        # Multiply user vector by item-item matrix to get prediction vector
        pred_vector = user_vector @ self.B

        # Get top-k items
        top_items = torch.topk(pred_vector, k, sorted=True).indices

        # Decode item IDs to original format
        item_ids = self.item_encoder.inverse_transform(top_items.cpu().numpy())

        return item_ids.tolist()


    def save(self, directory='/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/models'):
        # Save the model
        with open(os.path.join(directory, 'ease_model.pkl'), 'wb+') as f:
            pickle.dump(self, f)
        print('Model was successfully saved')

    @staticmethod
    def load(filename='/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/models/ease_model.pkl'):
        if not os.path.isfile(filename):
            raise Exception(f'File {filename} does not exist')
        # Load the model
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
