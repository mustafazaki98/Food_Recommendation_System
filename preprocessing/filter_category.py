# import pandas as pd
#
# # Load the CSV file
# df = pd.read_csv('final_data.csv')
# df_data = pd.read_csv('data.csv')
#
# # Filter out food IDs with purchase count less than 1 and missing food category
# df_filtered = df[~(df['purchase_count'] == 1) & (df['branded_food_category'].isna())]
# # df_filtered = df[(df['purchase_count'] == 1) & (df['Similarity'] > 0.05)]
#
# # Count the number of filtered food IDs
# num_filtered_food_ids = len(df) - len(df_filtered)
#
# # Print the filtered dataframe and the count of filtered food IDs
# print("Filtered Data:")
# print(df_filtered)
# df_filtered.to_csv('filtered.csv')
# print("\nNumber of filtered food IDs:", num_filtered_food_ids)
# print('Sum of NA Values', sum(df_filtered.drop_duplicates(subset='Product Name')['branded_food_category'].isna()))
# print('Sum of NA Values in df', sum(df.drop_duplicates(subset='Product Name')['branded_food_category'].isna()))
# # print('Sum of NA Values in df data', sum(df_data['branded_food_category'].isna()))


# import pandas as pd
#
# # Load the CSV file
# df = pd.read_csv('final_data.csv')
#
# # Filter out food IDs purchased only once
# print('Unique categories OG', df['branded_food_category'].nunique())
# similarity_threshold = 0.2
#
# filtered_categories = df[~df['branded_food_category'].isin(['Antipasto', 'Baking Accessories', 'Baking Needs', 'Baking/Cooking Supplies (Shelf Stable)', 'Cakes/Slices/Biscuits', 'Cheese - Block', 'Fresh Chicken - Processed', 'Fresh Fruit and Vegetables', 'Noodles', 'Pasta'])]
#
#
# print('Unique categories Filltered', filtered_categories['branded_food_category'].nunique())
#
#
# purchased_once = filtered_categories[filtered_categories['purchase_count'] == 1]
#
# # Filter out food IDs with missing category
# missing_category = filtered_categories[filtered_categories['branded_food_category'].isnull()]
#
# low_similarity = filtered_categories[filtered_categories['Similarity'] < similarity_threshold]
#
#
# # Filter the missing_category DataFrame
# missing_category_filtered = purchased_once.merge(missing_category, on='food_id', how='inner')
#
# # Filter the low_similarity DataFrame
# low_similarity_filtered = purchased_once.merge(low_similarity, on='food_id', how='inner')
#
# combined_filter = missing_category_filtered.merge(low_similarity, on='food_id', how='inner')
# # Get the count of filtered food IDs
# # filtered_count = len(filtered_df)
#
# low_similarity_indices = low_similarity_filtered.index
# missing_category_indices = missing_category_filtered.index
# combined_filter_indices = low_similarity_indices.intersection(missing_category_indices)
#
# # Delete the filtered rows from the original DataFrame
# deleted_items_df = df.loc[combined_filter_indices]
# df.drop(combined_filter_indices, inplace=True)
# # df.drop(missing_category_indices, inplace=True)
#
# df.to_csv('filtered.csv')
# deleted_items_df.to_csv('deleted_items.csv')
#
# # deleted_items_df = df.loc[low_similarity_indices]
#
# # Display the updated DataFrame
# print(df)
#
# # Display the count of filtered food IDs
# print('Low Similarity: ',len(low_similarity_indices))
# print('Missing Category: ',len(missing_category_indices))
# print(f"Number of food IDs purchased once with missing category and similarity < {similarity_threshold*100}%: ", len(low_similarity_indices) + len(missing_category_indices))
# print('Common number of values: ', len(combined_filter_indices))


### Method 2 ####

import pandas as pd

# Load the CSV file
# df = pd.read_csv('final_data.csv')
#
# # Filter out food IDs purchased only once
# # print('Unique categories OG', df['branded_food_category'].nunique())
# similarity_threshold = 0.2
#
# # Filter Categories
# filtered_categories = df[~df['branded_food_category'].isin(['Antipasto', 'Baking Accessories', 'Baking Needs', 'Baking/Cooking Supplies (Shelf Stable)', 'Cakes/Slices/Biscuits', 'Cheese - Block', 'Fresh Chicken - Processed', 'Fresh Fruit and Vegetables', 'Noodles', 'Pasta'])]
#
# print(df)
# # print('Unique categories Filltered', filtered_categories['branded_food_category'].nunique())
#
# # Purchased once (1)
# purchased_once = filtered_categories[filtered_categories['purchase_count'] == 1]
#
# # Delete NA Categories
# missing_category = filtered_categories[filtered_categories['branded_food_category'].isnull()]
# missing_category_filtered = purchased_once.merge(missing_category, on='food_id', how='inner')
# deleted_categories = df.drop(missing_category_filtered.index)
#
# # Purchased once (2)
# purchased_once2 = deleted_categories[deleted_categories['purchase_count'] == 1]
# low_similarity = deleted_categories[deleted_categories['Similarity'] < similarity_threshold]
# low_similarity_filtered = purchased_once2.merge(low_similarity, on='food_id', how='inner')
#
# combined_filter_df = deleted_categories.drop(low_similarity_filtered.index)

# combined_filter = missing_category_filtered.merge(low_similarity, on='food_id', how='inner')
# Get the count of filtered food IDs
# filtered_count = len(filtered_df)

# low_similarity_indices = low_similarity_filtered.index
# missing_category_indices = missing_category_filtered.index
# combined_filter_indices = low_similarity_indices.intersection(missing_category_indices)
#
# # Delete the filtered rows from the original DataFrame
# deleted_items_df = df.loc[combined_filter_indices]
# df.drop(combined_filter_indices, inplace=True)
# # df.drop(missing_category_indices, inplace=True)
#
# df.to_csv('filtered.csv')
# deleted_items_df.to_csv('deleted_items.csv')

# deleted_items_df = df.loc[low_similarity_indices]

# Display the updated DataFrame
# print(combined_filter_df)

# Display the count of filtered food IDs
# print('Low Similarity: ',len(low_similarity_indices))
# print('Missing Category: ',len(missing_category_indices))
# print(f"Number of food IDs purchased once with missing category and similarity < {similarity_threshold*100}%: ", len(low_similarity_indices) + len(missing_category_indices))
# print('Common number of values: ', len(combined_filter_indices))


### Method 3 ####


import pandas as pd

### Missing Categories ####

# Load the CSV file
# df = pd.read_csv('final_data_unique.csv')
#
# # Filter out food IDs purchased only once
# print('Unique categories OG', df['branded_food_category'].nunique())
#
# filtered_categories = df[~df['branded_food_category'].isin(['Antipasto', 'Baking Accessories', 'Baking Needs', 'Baking/Cooking Supplies (Shelf Stable)', 'Cakes/Slices/Biscuits', 'Cheese - Block', 'Fresh Chicken - Processed', 'Fresh Fruit and Vegetables', 'Noodles', 'Pasta'])]
#
#
# print('Unique categories Filltered', filtered_categories['branded_food_category'].nunique())
#
#
# purchased_once = filtered_categories[filtered_categories['purchase_count'] == 1]
#
# # Filter out food IDs with missing category
# missing_category = filtered_categories[filtered_categories['branded_food_category'].isnull()]
#
# missing_category_filtered = purchased_once.merge(missing_category, on='food_id', how='inner')
# missing_category_indices = missing_category_filtered.index
#
# print('Length of df Before: ',len(df))
# df.drop(missing_category_indices, inplace=True)
# print('Length of df After: ',len(df))
#
# missing_category_filtered.to_csv('deleted_items.csv',index=False)
#
# df.to_csv('filtered.csv',index=False)

#### Low Similarity ####

similarity_threshold = 0.1

filtered_categories=pd.read_csv('filtered.csv')

purchased_once = filtered_categories[filtered_categories['purchase_count'] == 1]


low_similarity = filtered_categories[filtered_categories['Similarity'] < similarity_threshold]

# Filter the low_similarity DataFrame
low_similarity_filtered = purchased_once.merge(low_similarity, on='food_id', how='inner')
low_similarity_indices = low_similarity_filtered.index

print('Length of df Before: ',len(filtered_categories))
filtered_categories.drop(low_similarity_indices, inplace=True)

print('Length of df After: ',len(filtered_categories))

print(filtered_categories)
print(low_similarity_filtered)

low_similarity_filtered.to_csv('deleted_items.csv', mode='a',header=False, index=False)

filtered_categories.to_csv('filtered.csv', index=False)
