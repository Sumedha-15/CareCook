import joblib
import json
import numpy as np
import pandas as pd
import os

# Load Advanced Model
MODEL_PATH = "health_safety_model_advanced.pkl"
THRESHOLDS_PATH = "condition_thresholds.json"

try:
    ai_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    ai_model = None

# Load Condition Thresholds
try:
    with open(THRESHOLDS_PATH, "r") as f:
        CONDITION_THRESHOLDS = json.load(f)
except FileNotFoundError:
    # Safe fallback if json is missing
    CONDITION_THRESHOLDS = {
        "none": {"sugar": 10, "fat": 20, "sodium": 300, "carbs": 50, "cholesterol": 300}
    }

def normalize_condition(medical_conditions):
    if not medical_conditions or medical_conditions.strip().lower() == "none":
        return ["none"]
    
    conditions =[c.strip().lower() for c in medical_conditions.split(",")]
    normalized =[]
    
    for cond in conditions:
        if "diabet" in cond: normalized.append("diabetes")
        elif "pcos" in cond or "pcod" in cond: normalized.append("pcos")
        elif "bp" in cond or "blood pressure" in cond or "hypertension" in cond: normalized.append("high bp")
        elif "heart" in cond or "cardiac" in cond: normalized.append("heart disease")
        elif "kidney" in cond or "renal" in cond: normalized.append("kidney disease")
        elif "obes" in cond: normalized.append("obesity")
        else: normalized.append(cond)
    
    return normalized

def get_strict_thresholds(conditions):
    if not conditions or conditions == ["none"]:
        return CONDITION_THRESHOLDS.get("none", {})
    
    merged = CONDITION_THRESHOLDS["none"].copy()
    for condition in conditions:
        if condition in CONDITION_THRESHOLDS:
            cond_threshold = CONDITION_THRESHOLDS[condition]
            for nutrient in merged.keys():
                merged[nutrient] = min(merged[nutrient], cond_threshold.get(nutrient, merged[nutrient]))
    return merged

def check_safety(row, condition):
    sugar = row.get("sugar", row.get("sugars", 0))
    fat = row.get("fat", 0)
    sodium = row.get("sodium", 0)
    carbs = row.get("carbs", row.get("carbohydrates", 0))
    cholesterol = row.get("cholesterol", 0)
    
    conditions = normalize_condition(condition)
    thresholds = get_strict_thresholds(conditions)
    
    passes_thresholds = (
        sugar <= thresholds.get('sugar', 100) and
        fat <= thresholds.get('fat', 100) and
        sodium <= thresholds.get('sodium', 1000) and
        carbs <= thresholds.get('carbs', 100) and
        cholesterol <= thresholds.get('cholesterol', 1000)
    )
    
    if ai_model is None:
        return 2 if passes_thresholds else 0
        
    features = pd.DataFrame({
        'Sugars': [sugar], 'Fat': [fat], 'Sodium': [sodium], 
        'Carbohydrates': [carbs], 'Cholesterol': [cholesterol]
    })
    ml_prediction = ai_model.predict(features)[0] 
    
    if passes_thresholds and ml_prediction >= 1: return 2
    elif passes_thresholds or ml_prediction >= 1: return 1
    else: return 0

def get_safety_explanation(safety_score, row, condition):
    conditions = normalize_condition(condition)
    thresholds = get_strict_thresholds(conditions)
    
    sugar = row.get("sugar", row.get("sugars", 0))
    fat = row.get("fat", 0)
    sodium = row.get("sodium", 0)
    carbs = row.get("carbs", row.get("carbohydrates", 0))
    cholesterol = row.get("cholesterol", 0)
    
    violations =[]
    if sugar > thresholds.get('sugar', 100): violations.append(f"Sugar {sugar}g > limit {thresholds['sugar']}g")
    if fat > thresholds.get('fat', 100): violations.append(f"Fat {fat}g > limit {thresholds['fat']}g")
    if sodium > thresholds.get('sodium', 1000): violations.append(f"Sodium {sodium}mg > limit {thresholds['sodium']}mg")
    if carbs > thresholds.get('carbs', 100): violations.append(f"Carbs {carbs}g > limit {thresholds['carbs']}g")
    if cholesterol > thresholds.get('cholesterol', 1000): violations.append(f"Cholesterol {cholesterol}mg > limit {thresholds['cholesterol']}mg")
    
    if violations: return f"Unsafe: {'; '.join(violations)}"
    return "Safe within your health constraints"