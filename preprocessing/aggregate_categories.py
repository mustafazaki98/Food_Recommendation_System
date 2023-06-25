# import pandas as pd
#
# def preprocess_category(category):
#     category = str(category)
#     return category.lower().replace('/', ' ')
#
# def aggregate_categories(df):
#     category_mapping = {}
#
#     for category in df['branded_food_category'].unique():
#         category = str(category)
#         preprocessed_category = preprocess_category(category)
#         words = preprocessed_category.split()
#         first_word = words[0]
#         if first_word not in category_mapping:
#             category_mapping[first_word] = preprocessed_category
#
#     # Apply category mapping to the dataframe
#     df['aggregated_category'] = df['branded_food_category'].apply(lambda x: category_mapping.get(str(preprocess_category(x)).split()[0], x))
#
#     return df
#
# # Load the CSV file
# df = pd.read_csv('final_data.csv')
#
# # Call the function to aggregate categories
# df = aggregate_categories(df)
# df.to_csv('aggregate.csv')
# # Print the resulting dataframe with aggregated categories
# print(df)




#### Similarity #####

import pandas as pd
from nltk import ngrams

def preprocess_category(category):
    return category.lower().replace('/', ' ')

def calculate_word_similarity(word1, word2):
    set1 = set(ngrams(word1, 1))
    set2 = set(ngrams(word2, 1))
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity = len(intersection) / len(union)
    return similarity

def aggregate_categories(df, similarity_threshold):
    category_mapping = {}

    for category in df['branded_food_category'].unique():
        category = str(category)
        preprocessed_category = preprocess_category(category)
        words = preprocessed_category.split()
        first_word = words[0]

        best_match = None
        best_similarity = 0

        for mapped_category in category_mapping:
            similarity = calculate_word_similarity(first_word, mapped_category)
            if similarity > best_similarity:
                best_match = mapped_category
                best_similarity = similarity

        if best_match is None or best_similarity < similarity_threshold:
            category_mapping[first_word] = preprocessed_category
        else:
            mapped_category = category_mapping[best_match]
            category_mapping[first_word] = mapped_category

    # Apply category mapping to the dataframe
    df['aggregated_category'] = df['branded_food_category'].apply(lambda x: category_mapping.get(preprocess_category(str(x)).split()[0], x))

    return df

# Load the CSV file
df = pd.read_csv('final_data.csv')

# Set the similarity threshold (adjust as per your needs)
similarity_threshold = 0.8

# Call the function to aggregate categories
df = aggregate_categories(df, similarity_threshold)
df.to_csv('aggregate.csv')
# Print the resulting dataframe with aggregated categories
print(df)
