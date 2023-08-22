from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer

health_data_path = 'Final Health Scores Normalized Linear.csv'
health_data = pd.read_csv(health_data_path)
food_df = pd.read_csv('/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/data/raw data/food.csv')

health_data = pd.merge(health_data, food_df, on='fdc_id', how='left')

# Encode descriptions using SentenceTransformer
print('Encoding Started')
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(health_data['description'].tolist())
print('\nEncoding Done')

# Calculate distortions (inertia) for different numbers of clusters
distortions = []
K_range = range(1, 150)
for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(embeddings)
    distortions.append(kmeans.inertia_)

# Plot the Elbow method
plt.figure(figsize=(10, 6))
plt.plot(K_range, distortions, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion (Inertia)')
plt.xticks(K_range)
plt.grid(True)
plt.show()
