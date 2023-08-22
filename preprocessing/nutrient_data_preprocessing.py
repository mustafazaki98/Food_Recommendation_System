import pandas as pd

food_nutrient = pd.read_csv('/Users/mustafazaki/Documents/Food_Recommendation_System/data/raw data/food_nutrient.csv')
nutrient_df = pd.read_csv('/Users/mustafazaki/Documents/Food_Recommendation_System/data/raw data/nutrient.csv')

grouped = food_nutrient.groupby('fdc_id')[['nutrient_id', 'amount']].apply(lambda x: dict(zip(x['nutrient_id'], x['amount']))).reset_index()
grouped.columns = ['fdc_id', 'nutrients']

nutrient_dict = dict(zip(nutrient_df['id'], nutrient_df['name'] + " (" + nutrient_df['unit_name'].str.lower() + ")"))

def replace_keys_with_names(nutrient_dict_local):
    return {nutrient_dict.get(k, k): v for k, v in nutrient_dict_local.items()}

grouped['nutrients'] = grouped['nutrients'].apply(replace_keys_with_names)

specified_nutrients = {
    'Protein (g)': 'Protein (g)',
    'Fatty acids, total saturated (g)': 'Fatty acids, total saturated (g)',
    'Sugars, total including NLEA (g)': 'Sugars, total (g)',
    'Vitamin C, total ascorbic acid (mg)': 'Vitamin C (mg)',
    'Potassium, K (mg)': 'Potassium (mg)',
    'Carbohydrate, by difference (g)': 'Carbohydrates (g)',
    'Sodium, Na (mg)': 'Sodium (mg)',
    'Fiber, total dietary (g)': 'Fiber, total dietary (g)',
}

for nutrient_key, column_name in specified_nutrients.items():
    grouped[column_name] = grouped['nutrients'].apply(lambda x: x.get(nutrient_key, 0))

grouped.drop(columns=['nutrients'], inplace=True)

output_path = "food_nutrients_transformed.csv"
grouped.to_csv(output_path, index=False)
