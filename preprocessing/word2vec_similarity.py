# import numpy as np
# from gensim.models import KeyedVectors
# from fuzzywuzzy import fuzz
# from gensim.models import Word2Vec
import csv
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Define dataset 1 and dataset 2 as lists of food items
user_df = pd.read_csv('/Users/mustafazaki/Downloads/Production-Tagger/user_data.csv')
products_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')

dataset1 = user_df['food_item']  # Replace with your dataset 1
dataset2 = products_df['Product Name'] # Replace with your dataset 2

# Replace NaN values with empty strings in dataset 1 and dataset 2
dataset1_cleaned = [item if isinstance(item, str) else '' for item in dataset1]
dataset2_cleaned = [item if isinstance(item, str) else '' for item in dataset2]
# Combine dataset 1 and dataset 2 into a single list
all_items = dataset1_cleaned + dataset2_cleaned

# Create TF-IDF vectors for all food items
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_items)

# Compute similarity scores and find the most similar item
results = []
for idx, item1 in enumerate(dataset1):
    vector1 = tfidf_matrix[idx]

    # Calculate cosine similarity between vector1 and all vectors in dataset 2
    similarity_scores = cosine_similarity(vector1, tfidf_matrix[len(dataset1):])
    max_similarity_index = similarity_scores.argmax()
    max_similarity_item = dataset2[max_similarity_index]
    max_similarity_score = similarity_scores[0, max_similarity_index]

    results.append([item1, max_similarity_item, max_similarity_score])
    print(f'{item1} ---------- {max_similarity_item} ---------- {max_similarity_score}')

# Save the results to a CSV file
output_file = 'similarity_results.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Extracted Data', 'Most Similar Item (USDA Dataset)', 'Similarity Score'])
    writer.writerows(results)

print(f"Similarity results have been saved to '{output_file}'.")



# Load the pre-trained Word2Vec model
# word2vec_model = Word2Vec.load('/Users/mustafazaki/Downloads/AgWordVectors-300.model')
#
# # Load dataset 1 and dataset 2
# user_df = pd.read_csv('/Users/mustafazaki/Downloads/Production-Tagger/user_data.csv')
# products_df = pd.read_csv('/Users/mustafazaki/Downloads/Food Recommendation System/data/preprocessed data/products.csv')
#
# dataset1 = user_df['food_item']  # Replace with your dataset 1
# dataset2 = products_df['Product Name']  # Replace with your dataset 2
#
# # Create a dictionary of dataset 2 items with their indices
# dataset2_dict = {item: idx for idx, item in enumerate(dataset2)}
#
# # Compute similarity scores and find the most similar item
# results = []
# for item1 in dataset1:
#     max_similarity_item = None
#     max_similarity_score = -1  # Initialize with a negative value
#
#     if item1 in word2vec_model:
#         vector1 = word2vec_model[item1]
#
#         for item2 in dataset2:
#             if item2 in word2vec_model:
#                 vector2 = word2vec_model[item2]
#                 similarity_score = cosine_similarity([vector1], [vector2])[0][0]
#
#                 if similarity_score > max_similarity_score:
#                     max_similarity_item = item2
#                     max_similarity_score = similarity_score
#
#     results.append([item1, max_similarity_item, max_similarity_score])
#     print(f'{item1} ---------- {max_similarity_item} ---------- {max_similarity_score}')
#
# # Save the results to a CSV file
# output_file = 'similarity_results.csv'
# with open(output_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['Dataset 1', 'Most Similar Item (Dataset 2)', 'Similarity Score'])
#     writer.writerows(results)
#
# print(f"Similarity results have been saved to '{output_file}'.")
