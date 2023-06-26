import os, sys
sys.path.append(os.path.abspath('/Users/mustafazaki/Downloads/Food Recommendation System/preprocessing'))
sys.path.append(os.path.abspath('/Users/mustafazaki/Downloads/Food Recommendation System/models'))

import numpy as np
import pandas as pd
from preprocessing import BinaryNormalizer
from ease import EASE, torch


# RANDOM_SEED = 42
# np.random.seed(RANDOM_SEED)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print('Device: ', device)

def main():
    binary_ease = BinaryNormalizer()
    preprocessed_df, names = binary_ease.fit_transform()


    # processed_df = data_preprocessing(df, names)
    ease_model = EASE(preprocessed_df)
    ease_model.fit()
    # ease_model.save()

    test_df = pd.DataFrame({
        'user_id': ['gib', 'gib', 'gib', 'kld', 'kld', 'kld', 'sds', 'sds', 'sds','1F2DD7D7-188B-4291-91F6-01CBA2845658','1F2DD7D7-188B-4291-91F6-01CBA2845658','1F2DD7D7-188B-4291-91F6-01CBA2845658','82BEF5A5-9A96-4F95-834D-B1648057FACC','82BEF5A5-9A96-4F95-834D-B1648057FACC','82BEF5A5-9A96-4F95-834D-B1648057FACC','82BEF5A5-9A96-4F95-834D-B1648057FACC','82BEF5A5-9A96-4F95-834D-B1648057FACC'],
        'food_id': ['u09132', 'u09050', 'u09316', 'u09003', 'u01117', 'u09050', 'g15547', 'b45252410', 'g16050','u01030','u01132','b45228316','b45255808','u11829','u22962','u11970','u20036'],
        'count': [12, 3, 4, 10, 5, 6, 4, 5, 8, 5, 5, 2,1,4,3,3,2]})
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
        print(f'\n--------------------- EASE: {user} ------------------------')
        # print(f'\n\nItems Purchased by the user: {user}')
        for purchased_item in test_df[test_df['user_id'] == user]['food_id']:
            print(
                f"{purchased_item} : {names.loc[purchased_item]['Product Name']}")
        print(f"\nItems Recommended to the user: {user}")
        for recommended_item in recommendations.loc[user]['recommendations']:
            print(
                f"{recommended_item} : {names.loc[recommended_item]['Product Name']}")


if __name__ == '__main__':
    main()
