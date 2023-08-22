import numpy as np
import pandas as pd
import math

import pandas as pd

class BinaryNormalizer:
    """
    A class for binary normalization of purchase data.

    Parameters:
    -----------
    data_df : pandas.DataFrame
        The raw purchase data containing 'user_id' and 'food_id' columns.

    Attributes:
    -----------
    raw_df : pandas.DataFrame
        The raw purchase data.
    purchase_count : numpy.ndarray
        The binary-normalized purchase data.
    """

    def __init__(self):
        """
        Initialize the BinaryNormalizer class.

        Parameters:
        -----------
        data_df : (pandas.DataFrame)
            The raw purchase data.
        """
        self.raw_df = pd.read_csv('/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/data/raw_data/final_user_data.csv', usecols=['user_id', 'food_id'], dtype={'food_id': 'object'})

        # self.raw_df = data_df

    def fit_transform(self):
        """
        Perform binary normalization on the purchase data.

        Returns:
        --------
        purchase_count : numpy.ndarray
            The binary-normalized purchase data.
        """
        df = self.raw_df.dropna()

        # Aggregate purchase data by user and food item.
        df = df.assign(count=1)
        df = df.groupby(['user_id', 'food_id'], as_index=False).agg({'count': 'sum'})

        # Convert purchase counts to binary values.
        df['count'] = df['count'].astype(bool).astype(int)

        return df



# class BinaryNormalizer:
#     """
#     A class for binary normalization of purchase data.
#
#     Parameters:
#     -----------
#     file_path : str
#         The path to the raw purchase data file.
#
#     Attributes:
#     -----------
#     raw_df : pandas.DataFrame
#         The raw purchase data.
#     names : pandas.DataFrame
#         The name of each food item.
#     purchase_count : numpy.ndarray
#         The binary-normalized purchase data.
#     """
#
#     def __init__(self):
#         """
#         Initialize the BinaryNormalizer class.
#
#         Parameters:
#         -----------
#         file_path : (str)
#             The path to the raw purchase data file.
#         """
#         # self.file_path = file_path
#         self.data_collection()
#
#     def data_collection(self):
#         """
#         Load the raw purchase data and food names from file.
#         """
#
#         self.raw_df = pd.read_csv('/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/preprocessing/final_user_data.csv', usecols=['user_id', 'food_id'], dtype={'most_similar_food_id': 'object'})
#         # self.raw_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data/recommender_data 1.csv', usecols=['user_id', 'food_id'])
#         # self.names = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')
#
#     def fit_transform(self):
#         """
#         Perform binary normalization on the purchase data.
#
#         Returns:
#         --------
#         purchase_count : numpy.ndarray
#             The binary-normalized purchase data.
#         """
#         df = self.raw_df.dropna()
#
#         # Aggregate purchase data by user and food item.
#         df = df.assign(count=df.groupby(['food_id', 'user_id'])['user_id'].transform('count'))
#         df.drop_duplicates(keep='first', inplace=True)
#         df = df.groupby(['user_id', 'food_id'], as_index=False).agg({'count': 'sum'})
#
#         # Convert purchase counts to binary values.
#         df['count'] = df['count'].astype(bool).astype(int)
#
#         # Merge with food item names and sort by purchase count.
#         # merged_df = pd.merge(df, self.names, how="inner", left_on="food_id", right_on="ID")
#         # merged_df.sort_values(by='count', ascending=False, inplace=True)
#
#         # # Convert purchase data to user-item matrix.
#         # user_item_matrix = merged_df.pivot(index='user_id', columns='food_id', values='count')
#         # user_item_matrix = user_item_matrix.fillna(0)
#         #
#         # # Convert user-item matrix to numpy array and calculate sparsity.
#         # self.purchase_count = user_item_matrix.values
#         # sparsity = 100 - (np.count_nonzero(self.purchase_count) / (self.purchase_count.shape[0] * self.purchase_count.shape[1])) * 100
#         # print('\nSparsity: {:.2f}%'.format(sparsity))
#
#         return df

class UnderSampler:
    def __init__(self, file_path='/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data'):
        self.file_path = file_path
        self.data_collection()

    def data_collection(self):
        """
        Collect the Data from:
        'recommender_data.csv' - stores number of items purchased by the user
        'products.csv' - stores Name of the Food Item with the corresponding Food Id

        """

        self.raw_df = pd.read_csv(self.file_path,
                         usecols=['user_id', 'food_id'])
        self.names = pd.read_csv(
            self.file_path)


    def fit_transform(self):
        """
        Performs Data Pre-processing

        Params:
        ====

        df : Data Frame
            Stores Data from 'recommender_data.csv'

        names: Data Frame
            Stores Data from 'products.csv'

        Returns
        ====
        purchase_count : (ndarray)
            user-item matrix with corresponding purchase count.

        """

        df = self.raw_df.dropna()
        df = df.assign(count=df.groupby(['food_id', 'user_id'])[
                       'user_id'].transform('count'))
        df.drop_duplicates(keep='first', inplace=True)

        df = df.groupby(['user_id', 'food_id'],
                        as_index=False).agg({'count': 'sum'})



        merged_df = pd.merge(df, self.names, how="inner",
                             left_on="food_id", right_on="ID")
        merged_df.sort_values(by='count', ascending=False, inplace=True)


        # print('User-Item Data Frame:\n')
        # print(merged_df.head())

        undersampled_filter = (
            merged_df['user_id'] == '94A4CACF-0F5F-4D95-811B-266719CD2FA9') | (merged_df['user_id'] == '1')
        undersampled_df = merged_df.loc[~undersampled_filter]


        count = undersampled_df['count']
        count_scaled = (count - count.min()) / (count.max() - count.min())

        normalised_df = undersampled_df.assign(playCountScaled=count_scaled)

        # user_item_matrix = normalised_df.pivot(
        #     index='user_id', columns='food_id', values='playCountScaled')
        #
        # user_item_matrix = user_item_matrix.fillna(0)
        #
        # self.purchase_count = user_item_matrix.values
        #
        # sparsity = float(len(purchase_count.nonzero()[0]))
        # sparsity /= (purchase_count.shape[0] * purchase_count.shape[1])
        # sparsity *= 100
        # print('\nSparsity: {:.2f}%'.format(100-sparsity))

        return normalised_df, self.names

class ZScoreNormaliser:
    """
    A class to preprocess purchase data for recommendation systems by normalizing purchase counts and
    creating a binary user-item matrix.

    Parameters:
    -----------
    file_path : str, optional (default='/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data')
        The file path to the raw data file.

    Attributes:
    -----------
    file_path : str
        The file path to the raw data file.
    raw_df : pandas.DataFrame
        The raw purchase data loaded from the raw data file.
    names : pandas.DataFrame
        The raw food item data loaded from the raw data file.
    purchase_count : numpy.ndarray
        A binary user-item matrix with normalized purchase counts.

    Methods:
    --------
    data_collection()
        Loads the raw purchase data and food item data from the file path.
    fit_transform(m=4)
        Normalizes the purchase counts and creates a binary user-item matrix.
        m : int, optional (default=4)
            The number of standard deviations from the mean purchase count above which purchases are considered high.
    """

    def __init__(self):
        """
        Initialize a BinaryNormaliser object.

        Parameters:
        -----------
        file_path : str, optional (default='/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data')
            The file path to the raw data file.
        """
        # self.file_path = file_path
        self.data_collection()

    def data_collection(self):
        """
        Load the raw purchase data and food item data from the file path.
        """
        self.raw_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data/recommender_data 1.csv', usecols=['user_id', 'food_id'])
        # self.raw_df = pd.read_csv('//Users/mustafazaki/Downloads/Food Recommendation System/preprocessing/new_user_data.csv', usecols=['user_id', 'food_id'])
        self.names = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')

    def fit_transform(self, m=4):
        """
        Normalize the purchase counts and create a binary user-item matrix.

        Parameters:
        -----------
        m : (int), default=4
            The number of standard deviations from the mean purchase count above which purchases are considered high.

        Returns:
        --------
        numpy.ndarray
            A normalized user-item matrix with purchase counts.
        """
        df = self.raw_df.dropna()

        # count purchases of each food item by each user
        df = df.assign(count=df.groupby(['food_id', 'user_id'])['user_id'].transform('count'))
        df.drop_duplicates(keep='first', inplace=True)
        df = df.groupby(['user_id', 'food_id'], as_index=False).agg({'count': 'sum'})

        # convert purchase counts to binary values
        # df['count'] = df['count'].astype(bool).astype(int)

        # merge purchase data with food item names
        merged_df = pd.merge(df, self.names, how="inner", left_on="food_id", right_on="ID")
        merged_df.sort_values(by='count', ascending=False, inplace=True)

        # identify high purchase counts
        mean_purchase_count = math.ceil(df['count'].mean())
        std_purchase_count = math.ceil(df['count'].std())
        threshold = mean_purchase_count + m * std_purchase_count

        # replace high purchase counts with the mean purchase count
        for user in df['user_id'].unique():
            high_purchase_foods = df[(df['user_id'] == user) & (df['count'] > threshold)]['food_id'].unique()
            for food in high_purchase_foods:
                df.loc[(df['user_id'] == user) & (df['food_id'] == food),
                       'count'] = mean_purchase_count

        merged_df = pd.merge(df, self.names, how="inner",
                             left_on="food_id", right_on="ID")
        merged_df.sort_values(by='count', ascending=False, inplace=True)

        count = merged_df['count']
        count_scaled = (count - count.min()) / (count.max() - count.min())

        normalised_df = merged_df.assign(playCountScaled=count_scaled)

        # user_item_matrix = merged_df.pivot(
        #     index='user_id', columns='food_id', values='playCountScaled')
        #
        # user_item_matrix = user_item_matrix.fillna(0)
        #
        # self.purchase_count = user_item_matrix.values
        #
        # sparsity = float(len(purchase_count.nonzero()[0]))
        # sparsity /= (purchase_count.shape[0] * purchase_count.shape[1])
        # sparsity *= 100
        # print('\nSparsity: {:.2f}%'.format(100-sparsity))

        return normalised_df, self.names

class LogNormaliser:
    """
    A class to preprocess purchase data for recommendation systems by normalizing purchase counts and
    creating a binary user-item matrix.

    Parameters:
    -----------
    file_path : str, optional (default='/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data')
        The file path to the raw data file.

    Attributes:
    -----------
    file_path : str
        The file path to the raw data file.
    raw_df : pandas.DataFrame
        The raw purchase data loaded from the raw data file.
    names : pandas.DataFrame
        The raw food item data loaded from the raw data file.
    purchase_count : numpy.ndarray
        A binary user-item matrix with normalized purchase counts.

    Methods:
    --------
    data_collection()
        Loads the raw purchase data and food item data from the file path.
    fit_transform(m=4)
        Normalizes the purchase counts and creates a binary user-item matrix.
        m : int, optional (default=4)
            The number of standard deviations from the mean purchase count above which purchases are considered high.
    """

    def __init__(self):
        """
        Initialize a BinaryNormaliser object.

        Parameters:
        -----------
        file_path : str, optional (default='/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data')
            The file path to the raw data file.
        """
        # self.file_path = file_path
        self.data_collection()

    def data_collection(self):
        """
        Load the raw purchase data and food item data from the file path.
        """
        # self.raw_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data/recommender_data 1.csv', usecols=['user_id', 'food_id'])
        self.raw_df = pd.read_csv('//Users/mustafazaki/Downloads/Food Recommendation System/preprocessing/new_user_data.csv', usecols=['user_id', 'food_id'])
        self.names = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')

    def fit_transform(self):
        """
        Normalize the purchase counts and create a binary user-item matrix.

        Returns:
        --------
        numpy.ndarray
            A normalized user-item matrix with purchase counts.
        """
        df = self.raw_df.dropna()

        # count purchases of each food item by each user
        df = df.assign(count=df.groupby(['food_id', 'user_id'])['user_id'].transform('count'))
        df.drop_duplicates(keep='first', inplace=True)
        df = df.groupby(['user_id', 'food_id'], as_index=False).agg({'count': 'sum'})

        # convert purchase counts to binary values
        df['count'] = np.log(df['count'])

        # merge purchase data with food item names
        merged_df = pd.merge(df, self.names, how="inner", left_on="food_id", right_on="ID")
        merged_df.sort_values(by='count', ascending=False, inplace=True)

        # identify high purchase counts
        # mean_purchase_count = math.ceil(df['count'].mean())
        # std_purchase_count = math.ceil(df['count'].std())
        # threshold = mean_purchase_count + m * std_purchase_count
        #
        # # replace high purchase counts with the mean purchase count
        # for user in df['user_id'].unique():
        #     high_purchase_foods = df[(df['user_id'] == user) & (df['count'] > threshold)]['food_id'].unique()
        #     for food in high_purchase_foods:
        #         df.loc[(df['user_id'] == user) & (df['food_id'] == food),
        #                'count'] = mean_purchase_count
        #
        # merged_df = pd.merge(df, self.names, how="inner",
        #                      left_on="food_id", right_on="ID")
        # merged_df.sort_values(by='count', ascending=False, inplace=True)
        #
        # count = merged_df['count']
        # count_scaled = (count - count.min()) / (count.max() - count.min())
        #
        # normalised_df = merged_df.assign(playCountScaled=count_scaled)

        # user_item_matrix = merged_df.pivot(
        #     index='user_id', columns='food_id', values='playCountScaled')
        #
        # user_item_matrix = user_item_matrix.fillna(0)
        #
        # self.purchase_count = user_item_matrix.values
        #
        # sparsity = float(len(purchase_count.nonzero()[0]))
        # sparsity /= (purchase_count.shape[0] * purchase_count.shape[1])
        # sparsity *= 100
        # print('\nSparsity: {:.2f}%'.format(100-sparsity))

        return merged_df, self.names
