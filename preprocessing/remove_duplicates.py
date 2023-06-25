import pandas as pd

# Load the CSV file
final_data = pd.read_csv('aggregate.csv')
user_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/raw data/recommender_data 1.csv')
# print('Original Number of food_id in user data', user_df['food_id'].nunique())
df = user_df[['user_id','food_id']].merge(final_data, left_on='food_id', right_on='food_id', how='left')
# print('Number of unique Items in final data', final_data.drop_duplicates(subset='Product Name')['Product Name'].nunique())
print(df['food_id'].nunique())
print(final_data)
# print(df)

# Drop duplicates based on food name and keep the first occurrence
df_unique = df.drop_duplicates(subset='Product Name', keep='first')

# Create a dictionary mapping food names to the first food ID
food_id_map = df_unique.set_index('Product Name')['food_id'].to_dict()

# Replace food IDs in the original DataFrame with the mapped food ID
df['food_id'] = df['Product Name'].map(food_id_map)
print(df['food_id'].nunique())
print(df)


# Create a file with user IDs and food IDs
df_user_food = df[['user_id', 'food_id']]
df_user_food.to_csv('new_user_data.csv', index=False)

# Create a file with unique food items
df_unique = df.drop_duplicates(subset='Product Name', keep='first')
df_unique = df_unique.drop(['user_id','Unnamed: 0'], axis=1)
print(df_unique)
df_unique.to_csv('final_data_unique.csv', index=False)

# final_data = final_data.merge(df['food_id'], left_on='food_id', right_on='food_id')
#
# print(sum(final_data['food_id'].isna()))

# Save the updated DataFrame to a new file
# df.to_csv('final_data_unique.csv', index=False)
