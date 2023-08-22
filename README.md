# Healthy Food Recommendation System 🍏🥗

In an era where personal well-being and nutrition are pivotal, the need for personalized and healthy food recommendations is more important than ever. This system not only caters to individual taste preferences but also ensures that the recommendations align with healthier choices.

This repository contains the code and data needed for a robust healthy food recommendation system. It analyzes users' meal descriptions to provide personalized recommendations, calculates health scores, and clusters food items to provide nutritious alternatives.

## 📁 Directory Structure

### **User Data**
- `Meal descriptions`: Descriptions of user meals
- `user_data`: Extracted user-related data

### **Food Items**
- `Food.csv`: Information about food items
- `food_nutrient.csv`: Nutrient data for food items
- `Nutrient.csv`: Detailed information about nutrients

## 🚀 Workflow

### **Part I: EASE (Personalized Recommendations)**
The aim of this part is to generate food recommendations that extend beyond the user's existing preferences, encouraging them to explore new items that might interest them.

1. **Preprocessing**
   - **Extract Meal Descriptions**: From meal descriptions file
   - **Generate User Data**: Using extracted meal descriptions → Output: `user_data.csv`
2. **Generate Recommendations**: Using `user_data.csv` → Food IDs for each user

### **Part II: Health Score Calculation**
1. **Preprocessing**
   - **Transform Nutrient Data**: Using `food_nutrients.csv` & `nutrient.csv` → Output: `food_nutrient_transformed.csv`
2. **Calculate Health Score**: For each food item in `food_nutrient_transformed.csv`

### **Part III: Clustering and Final Score**
1. **Create BERT Embeddings**: For food items
2. **Cluster Food Items**: Using BERT embeddings, store clusters in `foodClusters.pickle` and embeddings in `foodVectors.pickle`
3. **Retrieve Similar Items**: Use clusters to get similar items
4. **Calculate Similarity Score**: Using cosine similarity between embeddings
5. **Calculate Final Score**: Final Score = alpha * similarity_score + (1-alpha) * health_score, where alpha ranges from [0,1]. Alpha controls how healthy or similar the item should be.
