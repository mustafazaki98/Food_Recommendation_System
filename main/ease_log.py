import os, sys
sys.path.append(os.path.abspath('/Users/mustafazaki/Downloads/Food Recommendation System/preprocessing'))
sys.path.append(os.path.abspath('/Users/mustafazaki/Downloads/Food Recommendation System/models'))

import numpy as np
import pandas as pd
from preprocessing import LogNormaliser
from ease import EASE, torch


def main():
    log_ease = LogNormaliser()
    preprocessed_df, names = log_ease.fit_transform()

    # processed_df = data_preprocessing(df, names)
    ease_model = EASE(preprocessed_df)
    ease_model.fit()
    # ease_model.save()
    test_df = pd.DataFrame({
        'user_id': ['gib', 'gib', 'gib', 'kld', 'kld', 'kld', 'sds', 'sds', 'sds'],
        'food_id': ['u09132', 'u09050', 'u09316', 'u09003', 'u01117', 'u09050', 'g15547', 'b45252410', 'g16050'],
        'count': [12, 3, 4, 10, 5, 6, 4, 5, 8]})
    print('\nTest Dataframe: \n', test_df)

    recommendations = ease_model.predict(test_df, k=10)
    print('\n---------------------------------------------')
    print('\nRecommendations Dataframe: \n', recommendations)

    print('\n---------------------------------------------')
    # print('\nNames Dataframe: \n', names)
    names.set_index('ID', inplace=True)
    recommendations.set_index('user_id', inplace=True)
    # print(recommendations[0])
    # user, group_df = test_df.groupby('user_id')

    for user in recommendations.index:
        print('\n---------------------------------------------')
        print(f'\n\nItems Purchased by the user: {user}')
        for purchased_item in test_df[test_df['user_id'] == user]['food_id']:
            print(
                f"{purchased_item} : {names.loc[purchased_item]['Product Name']}")
        print(f"\nItems Recommended to the user: {user}")
        for recommended_item in recommendations.loc[user]['recommendations']:
            print(
                f"{recommended_item} : {names.loc[recommended_item]['Product Name']}")


if __name__ == '__main__':
    main()
