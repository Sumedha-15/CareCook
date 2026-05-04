import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
import glob

print("Looking for dataset files...")

# Find all CSV files that start with FOOD-DATA-GROUP
all_files = glob.glob("FOOD-DATA-GROUP*.csv")

if not all_files:
    raise FileNotFoundError("Could not find any FOOD-DATA-GROUP CSV files in this directory!")

print(f"Found {len(all_files)} datasets. Merging them now...")

# Merge all datasets into one giant dataframe
df_list =[pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# Save a master copy just in case you want to look at it later
df.to_csv("FOOD-DATA-MASTER.csv", index=False)
print(f"Merged successfully! Total ingredients to learn from: {len(df)}")

# ============== CONDITION-SPECIFIC THRESHOLDS ==============
CONDITION_THRESHOLDS = {
    "none": {"sugar": 10, "fat": 20, "sodium": 300, "carbs": 50, "cholesterol": 300},
    "diabetes": {"sugar": 3, "fat": 15, "sodium": 200, "carbs": 25, "cholesterol": 200},
    "pcos": {"sugar": 3, "fat": 12, "sodium": 180, "carbs": 20, "cholesterol": 150},
    "high bp": {"sugar": 5, "fat": 15, "sodium": 100, "carbs": 40, "cholesterol": 200},
    "heart disease": {"sugar": 2, "fat": 5, "sodium": 80, "carbs": 15, "cholesterol": 100},
    "kidney disease": {"sugar": 5, "fat": 10, "sodium": 50, "carbs": 30, "cholesterol": 150},
    "obesity": {"sugar": 2, "fat": 8, "sodium": 150, "carbs": 20, "cholesterol": 200},
    "jaundice": {"sugar": 10, "fat": 5, "sodium": 200, "carbs": 60, "cholesterol": 50}
}

# ============== CREATE MULTI-CONDITION LABELS ==============
print("Creating condition-aware labels for the AI...")

def calculate_safety_score(row):
    """Calculate how many conditions find this ingredient safe"""
    sugar = row.get('Sugars', 0)
    fat = row.get('Fat', 0)
    sodium = row.get('Sodium', 0)
    carbs = row.get('Carbohydrates', 0)
    cholesterol = row.get('Cholesterol', 0)
    
    safe_count = 0
    for condition, thresholds in CONDITION_THRESHOLDS.items():
        if (sugar <= thresholds['sugar'] and 
            fat <= thresholds['fat'] and 
            sodium <= thresholds['sodium'] and 
            carbs <= thresholds['carbs'] and 
            cholesterol <= thresholds['cholesterol']):
            safe_count += 1
    
    # Score: 2 = Very Safe, 1 = Moderately Safe, 0 = Unsafe
    if safe_count >= 6: return 2
    elif safe_count >= 2: return 1
    else: return 0

df['safety_score'] = df.apply(calculate_safety_score, axis=1)

# ============== SELECT FEATURES & SPLIT DATA ==============
features = ['Sugars', 'Fat', 'Sodium', 'Carbohydrates', 'Cholesterol']
X = df[features].fillna(0)
y = df['safety_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ============== TRAIN THE MODEL ==============
print("\nTraining Advanced Random Forest Model on thousands of ingredients...")
model = RandomForestClassifier(
    n_estimators=150, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

# ============== EVALUATE & SAVE ==============
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Final Model Accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "health_safety_model_advanced.pkl")
with open("condition_thresholds.json", "w") as f:
    json.dump(CONDITION_THRESHOLDS, f, indent=2)

print("✅ Saved 'health_safety_model_advanced.pkl'")
print("✅ Saved 'condition_thresholds.json'")
print("🎉 AI is successfully trained on all datasets! You can now run the app.")