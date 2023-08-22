import pandas as pd
import numpy as np

# Load the data
data_path = "/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/data/preprocessed data/food_nutrients_transformed.csv"
data = pd.read_csv(data_path)

data = data.rename(columns={'Sugars, total\n(g)': 'Sugars, total (g)'})


# Nutrients with their Daily Recommended Values
healthy_nutrients = {
    'Protein (g)': 60,  # g
    'Fiber, total dietary (g)': 28,  # g
    'Vitamin C (mg)': 90,  # mg
    'Potassium (mg)': 4700,  # mg
    # 'Total Fat (g)': 78 #g
}

dietary_recommendations = {
    'Protein (g)': [40, 300],  # g
    'Total Fat (g)': [70, 100],  # g
    'Sugars, total (g)': [6, 10],  # g
    'Fiber, total dietary (g)': [30, 100],  # g
    'Vitamin C (mg)': [75, 2000],  # mg
    'Potassium (mg)': [2700, 6000],  # mg
    'Carbohydrate (g)': [20, 40], #g
    'Fatty acids, total saturated (g)': 20,
    'Sodium (mg)':[400, 600] #mg
}

unhealthy_nutrients = {
    'Sugars, total (g)': 50,  # g
    'Carbohydrate (g)': 275, #g
    'Sodium (mg)': 2300, #mg
    'Fatty acids, total saturated (g)': 20,
}

# Weights for each nutrient
weights = {
    'Protein (g)': 1.2,  # g
    'Fatty acids, total saturated (g)': 1,  # g
    'Sugars, total (g)': 0.7,  # g
    'Fiber, total dietary (g)': 1,  # g
    'Vitamin C (mg)': 0.8,  # mg
    'Potassium (mg)': 0.8,  # mg
    'Carbohydrate (g)': 0.7, #g
    'Sodium (mg)': 0.7 #mg
}

# Function to calculate normalized health score
def calculate_normalized_health_score(row):
    score = 0

    # Maximising the Health Score based on healthy nutrients
    for nutrient, dv in healthy_nutrients.items():
        if nutrient not in row:
            continue

        x = row[nutrient]

        # Normalize x to percentage of recommended intake
        x /= (dv)

        y=x

        # Apply weight
        y *= weights[nutrient]

        score += y

    # Minimising the Health Score based on unhealthy nutrients
    for nutrient, dv in unhealthy_nutrients.items():
        if nutrient not in row:
            continue

        x = row[nutrient]

        x /= (dv)
        y= 0.05-x

        # Apply weight
        y *= weights[nutrient]

        score += y

    return score

# Calculate health scores for each row
data['Normalized Health Score'] = data.apply(calculate_normalized_health_score, axis=1)


# Calculate the IQR of the health scores
Q1 = data['Normalized Health Score'].quantile(0.25)
Q3 = data['Normalized Health Score'].quantile(0.75)
IQR = Q3 - Q1

# Define the outlier thresholds
lower_threshold = Q1 - 1.5 * IQR
upper_threshold = Q3 + 1.5 * IQR

# Extract the outliers
data_outliers = data[(data['Normalized Health Score'] < lower_threshold) | (data['Normalized Health Score'] > upper_threshold)]

# Remove the outliers
data_no_outliers = data[(data['Normalized Health Score'] >= lower_threshold) & (data['Normalized Health Score'] <= upper_threshold)]

# Calculate the correlation of each nutrient with the health score in the data with no outliers
correlations_no_outliers = data_no_outliers.corrwith(data_no_outliers['Normalized Health Score'])
correlations_no_outliers = correlations_no_outliers[[key for key in dietary_recommendations.keys() if key in correlations_no_outliers]]

a = -1
b = 1

# Minimum and maximum health scores
min_health_score = data['Normalized Health Score'].min()
max_health_score = data['Normalized Health Score'].max()

# Function to normalize the health score between -1 and 1
def normalize_health_score(score):
    return a + (score - min_health_score) * (b - a) / (max_health_score - min_health_score)

# Apply the normalization to the 'Normalized Health Score' column
data['Normalized Health Score (-1 to 1)'] = data['Normalized Health Score'].apply(normalize_health_score)

# Save the updated dataframe as a new CSV file
output_path_normalized = "/Users/jayadeepneerubavi/Downloads/Food_Recommendation_System/data/preprocessed data/Final Health Scores Normalized Linear.csv"
data.to_csv(output_path_normalized, index=False)
