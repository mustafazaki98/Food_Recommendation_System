# import pandas as pd
# from difflib import SequenceMatcher
#
# def calculate_similarity(a, b):
#     a = str(a)
#     b = str(b)
#     return SequenceMatcher(None, a, b).ratio()
#
# def combine_food_info(file1, file2, file3, output_file):
#     df1 = pd.read_csv(file1)
#     df2 = pd.read_csv(file2)
#     df3 = pd.read_csv(file3)
#
#     merged1_df = df1.merge(df3, on="fdc_id")
#     merged1_df = merged1_df.drop_duplicates()
#
#     output_rows = []
#
#     for index1, row1 in merged1_df.iterrows():
#         name1 = row1['description']
#         category1 = row1['branded_food_category']
#
#         for index2, row2 in df2.iterrows():
#             name2 = row2['Product Name']
#             # category2 = row2['Food Category']
#
#             similarity = calculate_similarity(name1, name2)
#             if similarity > 0.6:
#                 output_row = [name1, name2, category1]
#                 output_rows.append(output_row)
#                 break
#
#     output_df = pd.DataFrame(output_rows, columns=['Food Name 1', 'Food Name 2', 'Food Category'])
#     output_df.to_csv(output_file, index=False)
#
# # Usage example
# combine_food_info("/Users/mustafazaki/Downloads/branded apr 2023/food.csv", "/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv", "/Users/mustafazaki/Downloads/branded apr 2023/branded_food.csv", 'similar_food_info.csv')




#### TRYING WITH A SAMPLE FILE (ALL THE ITEMS ARE IN THE SAME FILE)

# import pandas as pd
# from difflib import SequenceMatcher
#
# def calculate_similarity(a, b):
#     a = str(a)
#     b = str(b)
#     return SequenceMatcher(None, a, b).ratio()
#
# def combine_food_info(file1, output_file):
#     df1 = pd.read_csv(file1)
#     # df2 = pd.read_csv(file2)
#     # df3 = pd.read_csv(file3)
#
#     # merged1_df = df1.merge(df3, on="fdc_id")
#     # merged1_df = merged1_df.drop_duplicates()
#     #
#     output_rows = []
#
#     # for index1, row1 in merged1_df.iterrows():
#     #     name1 = row1['description']
#     #     category1 = row1['branded_food_category']
#
#     for index2, row2 in df1.iterrows():
#         name1 = row2['Food Name 1']
#         name2 = row2['Food Name 2']
#         # category2 = row2['Food Category']
#
#         similarity = calculate_similarity(name1, name2)
#         output_row = [name1, name2, similarity]
#         output_rows.append(output_row)
#
#
#     output_df = pd.DataFrame(output_rows, columns=['Food Name 1', 'Food Name 2', 'Food Category'])
#     output_df.to_csv(output_file, index=False)
#
# # Usage example
# combine_food_info("/Users/mustafazaki/Downloads/similarity_test.csv", 'similar_food_info.csv')



####  TRTING WITH A SAMPLE FILE (ALL THE ITEMS ARE IN A DIFFERENT FILE - RUNNING A CROSS MERGE)

# import pandas as pd
# from difflib import SequenceMatcher
#
# def calculate_similarity(a, b):
#     return SequenceMatcher(None, a, b).ratio()
#
# def get_missing_items(file1, output_df, output_file):
#     df1 = pd.read_csv(file1)
#
#     file1_items = set(df1['Food Name 1'])
#     output_items = set(output_df['Food Name 1'])
#
#     missing_items = file1_items.difference(output_items)
#
#     missing_items_df = pd.DataFrame({'Missing Items': list(missing_items)})
#     missing_items_df.to_csv(output_file, index=False)
#
# def combine_food_info(file1, file2, output_file):
#     df1 = pd.read_csv(file1)
#     df2 = pd.read_csv(file2)
#
#     df1['key'] = 1
#     df2['key'] = 1
#
#     merged_df = pd.merge(df1, df2, on='key', how='outer')
#
#     merged_df['Similarity'] = merged_df.apply(lambda row: calculate_similarity(row['Food Name 1'], row['Food Name 2']), axis=1)
#     matching_rows = merged_df['Similarity'] > 0.6
#     output_df = merged_df[matching_rows]
#     output_failed_df = merged_df[~matching_rows]
#
#
#     output_df = output_df.drop('key', axis=1)
#     output_df = output_df.drop_duplicates()
#
#     output_df.to_csv(output_file, index=False)
#
#     # Usage example
#     get_missing_items(file1, output_df, 'missing_items.csv')
#
#
# combine_food_info('/Users/mustafazaki/Downloads/food1.csv', '/Users/mustafazaki/Downloads/food2.csv', 'cross_merge_file.csv')


#### RUNNING THE ABOVE CODE ON THE ACTUAL DATA

import pandas as pd
from difflib import SequenceMatcher

def calculate_similarity(str1, str2):
    str1 = str(str1)
    str2 = str(str2)
    set1 = set(str1.split())
    set2 = set(str2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity


def get_missing_items(file1, output_df, output_file):
    df1 = pd.read_csv(file1)

    file1_items = set(df1['Food Name'])
    output_items = set(output_df['Food Name 1'])

    missing_items = file1_items.difference(output_items)

    missing_items_df = pd.DataFrame({'Missing Items': list(missing_items)})
    missing_items_df.to_csv(output_file, index=False)

def combine_food_info(file1, file2, file3,  output_file):

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    merged1_df = df1.merge(df2, left_on="food_id", right_on='ID')
    df1 = merged1_df.drop_duplicates()

    # df1['key'] = 1
    df3['key'] = 1

    matching_dfs = []
    unmatched_dfs = []

    batch_size = 102

    # print(df1)
    # print(df3)

    # print(df1['Product Name'].nunique())
    unique_items_df = pd.DataFrame({'Product Name': df1['Product Name'].unique()})
    unique_items_df['key'] = 1

    food_df = pd.DataFrame({'Food Name': df3['description'].unique()})
    food_df['key'] = 1

    print(len(food_df))
    print(len(unique_items_df))

    for i in range(0, len(unique_items_df), batch_size):
        batch_df1 = unique_items_df.iloc[i:i+batch_size]
        merged_df = pd.merge(batch_df1, food_df, how='outer', on='key')
        merged_df['Similarity'] = merged_df.apply(lambda row: calculate_similarity(row['Product Name'], row['Food Name']), axis=1)
        output_df = merged_df[merged_df['Similarity'] >= 0.5].drop_duplicates()
        output_df = output_df.groupby('Food Name').apply(lambda group: group.loc[group['Similarity'].idxmax()]).reset_index(drop=True)
        matching_dfs.append(output_df)
        print(f'------- Batch {i} Done -------')

    # for i in range(0, len(unique_items_df), batch_size):
    #     batch_df1 = unique_items_df.iloc[i:i+batch_size]
    #     merged_df = pd.merge(batch_df1, food_df, on='key', how='outer')
    #     merged_df = merged_df.drop_duplicates()
    #     merged_df['Similarity'] = merged_df.apply(lambda row: calculate_similarity(row['Product Name'], row['Food Name']), axis=1)
    #
    #     matching_rows = merged_df['Similarity'] > 0.6
    #     output_df = merged_df[matching_rows]
    #
    #     matching_dfs.append(output_df.drop_duplicates())
    #
    #     print(f'------- Batch {i} Done -------')

    matched_df = pd.concat(matching_dfs)

    # matching_rows = merged_df['Similarity'] > 0.5
    # final_df = merged_df[matching_rows]
    # output_failed_df = merged_df[~matching_rows]


    # output_df = output_df.drop('key', axis=1)
    final_df = matched_df.drop_duplicates()

    final_df.to_csv(output_file, index=False)

    # get_missing_items(file1, matched_df, 'missing_items.csv')


combine_food_info("non_matching_user_items.csv", "/Users/jayadeepneerubavi/Downloads/Food Recommendation System/data/preprocessed data/products.csv", "/Users/jayadeepneerubavi/Downloads/Food Recommendation System/data/raw data/food.csv", 'similar_food_info.csv')




##### TRYING EFFICIENT WAY

# import pandas as pd
# from difflib import SequenceMatcher
#
# def calculate_similarity(a, b):
#     a = str(a)
#     b = str(b)
#     return SequenceMatcher(None, a, b).ratio()
#
# def get_missing_items(file1, output_df, output_file):
#     df1 = pd.read_csv(file1)
#
#     file1_items = set(df1['Food Name 1'])
#     output_items = set(output_df['Food Name 1'])
#
#     missing_items = file1_items.difference(output_items)
#
#     missing_items_df = pd.DataFrame({'Missing Items': list(missing_items)})
#     missing_items_df.to_csv(output_file, index=False)
#
# def combine_food_info(file1, file2, file3, output_file):
#     df1 = pd.read_csv(file1)
#     df2 = pd.read_csv(file2)
#     df3 = pd.read_csv(file3)
#
#     merged1_df = df1.merge(df3, left_on='Food Name 1', right_on='description')
#     merged1_df = merged1_df.drop_duplicates()
#
#     output_rows = []
#     batch_size = 102
#
#     for i in range(0, len(merged1_df), batch_size):
#         batch_df = merged1_df.iloc[i:i+batch_size]
#         names1 = batch_df['Food Name 1'].tolist()
#         categories1 = batch_df['branded_food_category'].tolist()
#
#         for name1, category1 in zip(names1, categories1):
#             max_similarity = 0
#             max_name2 = ''
#             for _, row in df2.iterrows():
#                 name2 = row['Food Name 2']
#                 similarity = calculate_similarity(name1, name2)
#                 if similarity > max_similarity:
#                     max_similarity = similarity
#                     max_name2 = name2
#
#             if max_similarity > 0.6:
#                 output_rows.append([name1, max_name2, category1])
#
#     output_df = pd.DataFrame(output_rows, columns=['Food Name 1', 'Food Name 2', 'Food Category'])
#     output_df.to_csv(output_file, index=False)
#
#     get_missing_items(file1, output_df, 'missing_items.csv')

# Usage example
# combine_food_info("non_matching_user_items.csv", "/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv", "/Users/mustafazaki/Downloads/branded apr 2023/food.csv", 'similar_food_info.csv')





#### GENERATING UNIQUE FOOD ITEMS

# import pandas as pd
#
# def save_unique_items_to_csv(user_df, products_df, output_file):
#
#     df1 = user_df
#     df2 = products_df
#     column_name = 'Product Name'
#
#     print('Unique Items in user df: ', user_df['food_id'].nunique())
#
#     df = pd.merge(df1, df2, left_on='food_id', right_on='ID', how='inner')
#     df = df.drop_duplicates()
#
#     print('Unique Items in df: ', df['food_id'].nunique())
#
#     unique_items = df[column_name].unique()
#     unique_items_df = pd.DataFrame({column_name: unique_items})
#     unique_items_df.to_csv(output_file, index=False)
#
#     print('Unique Items: ', unique_items_df[column_name].nunique())
#
# # Usage example
# user_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data/recommender_data 1.csv')
# products_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')
# save_unique_items_to_csv(user_df, products_df, 'unique_food_items.csv')
