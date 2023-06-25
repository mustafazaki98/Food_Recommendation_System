import pandas as pd

user_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data/recommender_data 1.csv')
similar_df = pd.read_csv('similar_food_info 2.csv')
# food_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')
# matching_df = pd.read_csv('matching_items.csv')
branded_df = pd.read_csv("/Users/mustafazaki/Downloads/branded apr 2023/branded_food.csv")
food_df = pd.read_csv('/Users/mustafazaki/Downloads/branded apr 2023/food.csv')
df = pd.read_csv('data.csv')
food_count = pd.read_csv('food_counts.csv')
# unique_food = pd.DataFrame({'food_id': user_df['food_id'].unique()})
# print(user_df['user_id'].dropna().nunique())
#
# food_items = unique_food.merge(food_df, left_on = 'food_id', right_on = 'ID', how = 'left')
# food_items = food_items.drop_duplicates()
# food_items = food_items.drop('ID', axis=1)
#
# print(food_items['Product Name'].nunique())
#
# food_items.to_csv('data.csv')
print(len(similar_df))

similar_df = similar_df.merge(food_df[['description','fdc_id']], left_on='Food Name', right_on='description', how='left')
similar_df = similar_df.drop_duplicates(subset='Product Name')

# print(sum(similar_df['fdc_id'].isna()))
# similar_df.to_csv('categories.csv',index=False)

categories_df = similar_df.merge(branded_df, left_on='fdc_id', right_on='fdc_id', how='left')
categories_df = categories_df.drop_duplicates()
categories_df = categories_df[['Product Name', 'Food Name', 'Similarity', 'branded_food_category', 'fdc_id', 'ingredients']]

# print(len(categories_df[categories_df['Similarity']<=0.3]))

final_df = categories_df.merge(df, left_on='Product Name', right_on='Product Name', how='left')
final_df = final_df.drop_duplicates()

final_df = final_df.merge(food_count, left_on='food_id', right_on='food_id', how='left')
final_df = final_df.drop_duplicates()


print(final_df)
print(sum(final_df['purchase_count'].isna()))

final_df.to_csv('final_data.csv',index=False)


# print(sum(categories_df['branded_food_category'].isna()))
# print(sum(categories_df['ingredients'].isna()))
# print(sum(categories_df['branded_food_category'].isna()))

# print(categories_df)

#### END ####

# print(similar_df['Product Name'].nunique())
# print(similar_df['Food Name'].nunique())

# data_df = data_df.drop_duplicates(subset='food_id')
# print(len(data_df))
#
# # Remove duplicates from the right dataframe based on 'food_name'
# print(len(products_df))
# products_df = products_df.drop_duplicates(subset='description')
# # products_df['fdc_id'] = products_df['fdc_id'].astype(str)
# print(len(products_df))
#
# merged_df = data_df.merge(products_df, left_on='Product Name', right_on='description', how='left', indicator=True)
#
# unmatched_df = merged_df[merged_df['_merge'] == 'left_only']
# unmatched_df = unmatched_df.drop_duplicates()
#
# num_unmatched_rows = len(unmatched_df)
#
# unmatched_df.to_csv('unmatched_rows.csv', index=False)
#
# print(f"Number of unmatched rows: {num_unmatched_rows}")
#
# similar_df.to_csv('similar.csv')
