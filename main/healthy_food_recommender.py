import os, sys
sys.path.append(os.path.abspath('/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/preprocessing'))
sys.path.append(os.path.abspath('/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/models'))

import numpy as np
import pandas as pd
from preprocessing import BinaryNormalizer
from ease import EASE, torch

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json

class HealthyFoodRecommender:
    def __init__(self, user_id, model='EASE', alpha=0.5):
        self.health_data_path = '/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/data/preprocessed data/Final Health Scores Normalized Linear.csv'
        self.embedding_file_path = "/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/data/preprocessed data/foodVectors.pickle"
        self.cluster_file_path = "/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/data/preprocessed data/foodClusters.pickle"
        self.ease_model_path = '/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/models/ease_model.pkl'
        self.user_id = user_id
        self.model = model
        self.alpha = alpha

        self.load_model()
        self.load_data()
        self.load_or_generate_clusters()

    def load_model(self):
        if self.model == 'EASE':
            if os.path.exists(self.ease_model_path):
                ease_model = EASE.load()
                self.recommendations = ease_model.predict(self.user_id, k=15)
                print(self.recommendations)

            else:
                binary_ease = BinaryNormalizer()
                preprocessed_df = binary_ease.fit_transform()
                ease_model = EASE(preprocessed_df)
                ease_model.fit()
                ease_model.save()

                self.recommendations = ease_model.predict(self.user_id, k=15)
                print(self.recommendations)

    def load_data(self):
        self.health_data = pd.read_csv(self.health_data_path)
        self.food_df = pd.read_csv('/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/data/raw data/food.csv')
        self.health_df = pd.merge(self.health_data, self.food_df, on='fdc_id', how='left').drop_duplicates()
        self.health_data = self.health_df[['fdc_id', 'description', 'Normalized Health Score (-1 to 1)']]

    def load_or_generate_clusters(self):
        # Check if embeddings and clusters files exist
        if os.path.exists(self.embedding_file_path) and os.path.exists(self.cluster_file_path):
            with open(self.embedding_file_path, 'rb') as f:
                self.embeddings = pickle.load(f)

            with open(self.cluster_file_path, 'rb') as f:
                self.clustered = pickle.load(f)
            print("Loaded existing embeddings and clusters.")
        else:
            # Generate embeddings and clusters if files do not exist
            print("Generating embeddings and clusters...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings = model.encode(self.health_data['description'].tolist())
            kmeans = KMeans(n_clusters=150)
            self.clustered = kmeans.fit_predict(self.embeddings)
            # Save the generated embeddings and clusters
            with open(self.embedding_file_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            with open(self.cluster_file_path, 'wb') as f:
                pickle.dump(self.clustered, f)
            print("Saved new embeddings and clusters.")


    def get_healthy_alternatives(self, k=16):
        food_ids = self.recommendations
        user_alternatives = []

        for food_id in food_ids:
            food_alternatives = {
                "food_id": food_id,
                "food name": self.health_data[self.health_data['fdc_id']==int(food_id)]['description'].iloc[0],
                "Health Score": self.health_data[self.health_data['fdc_id']==int(food_id)]['Normalized Health Score (-1 to 1)'].iloc[0],
                "alternatives": []
            }

            # Get the healthy alternatives
            similar_foods = self.get_similar_foods_in_cluster(food_id, k)
            print(similar_foods)

            for idx, food in similar_foods.iterrows():
                alternative = {
                    "food_id": food['fdc_id'],
                    "description": food['Food Description'],
                    "health_score": food['Health Score']
                }
                food_alternatives["alternatives"].append(alternative)

            user_alternatives.append(food_alternatives)

        return user_alternatives


    def get_similar_foods_in_cluster(self, fdc_id, k):
        print(f"Type of fdc_id: {type(fdc_id)}")
        print(f"Type of 'fdc_id' column in DataFrame: {self.health_data['fdc_id'].dtype}")
        fdc_id = int(fdc_id)
        food_index = self.health_data[self.health_data['fdc_id'] == fdc_id].index[0]
        k = 16
        print("\nSelected Food:")
        print(self.health_data['description'].iloc[food_index])
        print(self.health_data['Normalized Health Score (-1 to 1)'].iloc[food_index])
        alpha = self.alpha

        cluster_label = self.clustered[food_index]

        # Get the indices of all the food items in the same cluster
        cluster_indices = np.where(self.clustered == cluster_label)[0]

        # Retrieve the embeddings and health scores for the food items in the cluster
        cluster_embeddings = self.embeddings[cluster_indices]
        cluster_health_scores = self.health_data['Normalized Health Score (-1 to 1)'].iloc[cluster_indices].values

        # Calculate the cosine similarity matrix for the embeddings in the cluster
        similarity_matrix = cosine_similarity(cluster_embeddings)

        cluster_food_index = np.where(cluster_indices == food_index)[0][0]

        for i in range(similarity_matrix.shape[0]):
            similarity_matrix[cluster_food_index][i] = alpha * similarity_matrix[cluster_food_index][i] + (1 - alpha) * cluster_health_scores[i]

        top_indices = similarity_matrix[cluster_food_index].argsort()[-k-1:-1][::-1]

        similar_foods = pd.DataFrame({
            'fdc_id': self.health_data['fdc_id'].iloc[cluster_indices[top_indices]],
            'Food Description': self.health_data['description'].iloc[cluster_indices[top_indices]],
            'Health Score': cluster_health_scores[top_indices],
            'Final Score': similarity_matrix[cluster_food_index, top_indices]
        })

        return similar_foods.sort_values(by='Health Score',ascending=False)[similar_foods['Health Score']>self.health_data['Normalized Health Score (-1 to 1)'][food_index]]


    def save_to_json(self, healthy_alternatives, output_file):
        with open(output_file, 'w') as file:
            json.dump(healthy_alternatives, file)

recommender = HealthyFoodRecommender('oryoo')

healthy_alternatives = recommender.get_healthy_alternatives()
recommender.save_to_json(healthy_alternatives, 'healthy_alternatives.json')
