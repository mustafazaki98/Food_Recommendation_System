import pandas as pd
import difflib

# Load the CSV file
df = pd.read_csv('final_data_unique.csv')

# Function to calculate similarity between two strings using the Levenshtein distance
def similarity_score(str1, str2):
    str1 = str(str1)
    str2 = str(str2)
    return difflib.SequenceMatcher(None, str1, str2).ratio()

df['similarity_score'] = 0.0

for i in range(len(df)):
    for j in range(i + 1, len(df)):
        sim_score = similarity_score(df['Product Name'][i], df['Product Name'][j])
        df.at[i, 'similarity_score'] = max(df.at[i, 'similarity_score'], sim_score)
        df.at[j, 'similarity_score'] = max(df.at[j, 'similarity_score'], sim_score)
    # print(f'Finished Batch {i}')

# Filter out rows with similarity greater than 0.9 and keep only the first occurrence
df_filtered = df[df['similarity_score'] <= 0.9].drop_duplicates(subset='Product Name', keep='first')

df_deleted = df[df['similarity_score'] > 0.9]

# Remove the 'similarity' column
df_filtered = df_filtered.drop('similarity_score', axis=1)

# Save the filtered data to a new CSV file
df_deleted.to_csv('delete_similar_data.csv', index=False)
