import os, json, glob, re, traceback
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from model_logic import check_safety, get_safety_explanation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "carecook-dev-secret-2024")
CORS(app)

GROQ_KEY     = os.getenv("GROQ_KEY")
UNSPLASH_KEY = os.getenv("UNSPLASH_KEY", "")
RECIPES_FILE = os.path.join(BASE_DIR, "saved_recipes.json")

print(f"[startup] GROQ_KEY loaded: {'YES' if GROQ_KEY else 'NO'}")
groq_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

def call_groq(messages):
    """Call Groq API with fallback to supported chat completion models."""
    if not groq_client:
        raise ValueError("GROQ_KEY not configured")
    
    candidate_models = [
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant"
    ]
    env_model = os.getenv("GROQ_MODEL")
    if env_model:
        candidate_models.insert(0, env_model)
        
    last_err = None
    for model in candidate_models:
        try:
            res = groq_client.chat.completions.create(
                model=model,
                messages=messages
            )
            return res.choices[0].message.content
        except Exception as e:
            last_err = e
            print(f"[groq fallback] Model '{model}' failed: {e}")
            continue
    raise last_err

# ── Food data ─────────────────────────────────────────────────────────────────
def load_food_data():
    search_patterns = [
        os.path.join(BASE_DIR, "data", "FOOD-DATA-MASTER.csv"),
        os.path.join(BASE_DIR, "FOOD-DATA-GROUP*.csv"),
        "data/FOOD-DATA-MASTER.csv",
        "FOOD-DATA-GROUP*.csv"
    ]
    paths = []
    for pat in search_patterns:
        paths.extend(glob.glob(pat))
    paths = list(dict.fromkeys(paths))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df.columns = df.columns.str.lower()
    print(f"[startup] Food data loaded: {len(df)} rows")
    return df

FOOD_DF = load_food_data()

# ── Matching config ───────────────────────────────────────────────────────────
# Words that make a food entry less desirable (processed/composite/branded)
BAD_WORDS = (
    "fried|canned|sauce|syrup|soup|paste|stewed|powder|rings|chips|"
    "dried|frozen|imitation|gravy|sandwich|nugget|burger|mcdonalds|kfc|"
    "wendys|crispy|strip|wing|skin|gizzard|heart|liver|feet|neck|back|"
    "bologna|patty|flavored|casserole|quesadilla|kung|lemon|tso|chow|"
    "pot pie|spread|stock|broth|bouillon|meatless|smoked|cured|"
    "instant|ready|blend|bowl|box|meal|roll|wrap|taco|pizza|hash|"
    "stuffed|loaded|deluxe|premium|breaded|battered|snack|sweet sour|"
    "flour|bread|juice|extract|concentrate"
)

# Prefer cooked/boiled/steamed — raw values for grains are dry-weight (inflated)
GOOD_WORDS  = "cooked|boiled|steamed|fresh|plain|whole"
RAW_PENALTY = "raw"

def best_match(item, df):
    """Pick the most realistic (cooked) and least processed match."""
    # 1. Exact match
    exact = df[df["food"] == item]
    if not exact.empty:
        return exact.iloc[0]

    # 2. Word-boundary search
    pattern = r"\b" + re.escape(item) + r"\b"
    hits = df[df["food"].str.contains(pattern, na=False, case=False, regex=True)].copy()

    # 3. Substring fallback
    if hits.empty:
        hits = df[df["food"].str.contains(re.escape(item), na=False, case=False)].copy()

    if hits.empty:
        return None

    # _bad:  1 = processed/junk name
    # _raw:  1 = "raw" in name (dry-weight, inflated carbs/fat)
    # _good: 0 = cooked/boiled/steamed (realistic values)
    # _len:  shorter = more generic
    hits["_bad"]  = hits["food"].str.contains(BAD_WORDS, case=False, regex=True).astype(int)
    hits["_raw"]  = hits["food"].str.contains(RAW_PENALTY, case=False, regex=True).astype(int)
    hits["_good"] = (~hits["food"].str.contains(GOOD_WORDS, case=False, regex=True)).astype(int)
    hits["_len"]  = hits["food"].str.len()

    # Sort: non-processed → non-raw → cooked/plain → shorter name
    hits = hits.sort_values(["_bad", "_raw", "_good", "_len"])
    return hits.iloc[0]

# ── Condition-specific food bans ──────────────────────────────────────────────
# These ingredients are medically unsafe for specific conditions
# regardless of what the ML threshold says
CONDITION_BANS = {
    "jaundice": [
        "chicken", "meat", "beef", "pork", "mutton", "lamb", "fish", "egg",
        "butter", "cream", "cheese", "milk", "ghee", "oil", "fat",
        "alcohol", "coffee", "tea", "soda", "fried", "spicy", "pickle"
    ],
    "kidney disease": [
        "banana", "orange", "potato", "tomato", "spinach", "salt",
        "pickle", "nuts", "chocolate", "dairy", "avocado"
    ],
    "heart disease": [
        "butter", "ghee", "cream", "bacon", "sausage", "salt", "pickle",
        "coconut oil", "palm oil", "red meat", "fried"
    ],
}

def is_banned_for_condition(item, condition):
    """Return True if this ingredient is medically banned for the condition."""
    conds = [c.strip().lower() for c in condition.split(",")] if condition else []
    for cond in conds:
        bans = CONDITION_BANS.get(cond, [])
        for ban in bans:
            if ban in item.lower() or item.lower() in ban:
                return True, cond
    return False, None

# ── Ingredient safety analysis ────────────────────────────────────────────────
def analyse_ingredients(ingredient_str, condition):
    items = [i.strip().lower() for i in ingredient_str.split(",") if i.strip()]
    safe, removed = [], []

    for item in items:
        # Check medical ban list first
        banned, ban_cond = is_banned_for_condition(item, condition)
        if banned:
            removed.append({
                "name": item, "matched_food": None, "nutrition": {},
                "score": 0, "label": "unsafe",
                "explanation": f"Medically unsafe for {ban_cond} — should be completely avoided"
            })
            print(f"[analyse] '{item}' -> BANNED for {ban_cond}")
            continue

        matched = None
        if not FOOD_DF.empty:
            matched = best_match(item, FOOD_DF)

        if matched is not None:
            nutrition = {
                "sugar":       float(matched.get("sugars", 0) or 0),
                "fat":         float(matched.get("fat", 0) or 0),
                "sodium":      float(matched.get("sodium", 0) or 0),
                "carbs":       float(matched.get("carbohydrates", 0) or 0),
                "cholesterol": float(matched.get("cholesterol", 0) or 0),
            }
            score       = check_safety(nutrition, condition)
            explanation = get_safety_explanation(score, nutrition, condition)
            row = {
                "name":         item,
                "matched_food": str(matched.get("food", item)),
                "nutrition":    nutrition,
                "score":        int(score),
                "label":        "safe" if score >= 2 else "caution" if score == 1 else "unsafe",
                "explanation":  explanation,
            }
            print(f"[analyse] '{item}' -> '{matched.get('food')}' fat={nutrition['fat']:.1f} sodium={nutrition['sodium']:.0f} score={score}")
            if score >= 1:
                safe.append(row)
            else:
                removed.append(row)
        else:
            print(f"[analyse] '{item}' -> not in DB, assumed safe")
            safe.append({
                "name": item, "matched_food": None, "nutrition": {},
                "score": 2, "label": "safe", "explanation": "Not found in DB — assumed safe"
            })

    return safe, removed

# ── Condition-specific prompt rules ───────────────────────────────────────────
CONDITION_RULES = {
    "jaundice": (
        "JAUNDICE STRICT RULES: The liver is damaged. "
        "ABSOLUTELY NO meat, chicken, fish, eggs, dairy, butter, ghee, oil, or any fatty/fried food. "
        "ONLY suggest light vegetarian dishes: rice, dal, boiled vegetables, fruits. "
        "If any safe ingredient is still heavy for the liver, exclude it."
    ),
    "kidney disease": (
        "KIDNEY DISEASE STRICT RULES: "
        "Avoid high potassium (banana, potato, tomato, spinach), high phosphorus (dairy, nuts), "
        "and high sodium foods. Only low-potassium, low-phosphorus, low-sodium dishes."
    ),
    "heart disease": (
        "HEART DISEASE STRICT RULES: "
        "No saturated fats, no trans fats, no high sodium. "
        "Prefer omega-3 rich, fibre-rich, low-cholesterol dishes."
    ),
    "diabetes": (
        "DIABETES STRICT RULES: "
        "No high-sugar or high-carb foods. Low glycemic index only. "
        "Prefer high-fibre, high-protein, low-carb dishes."
    ),
    "high bp": (
        "HIGH BP STRICT RULES: "
        "No salt, no high-sodium foods, no processed foods. "
        "Prefer potassium-rich, magnesium-rich, low-sodium dishes."
    ),
}

def get_condition_rules(condition):
    if not condition or condition == "none":
        return ""
    rules = []
    for cond, rule in CONDITION_RULES.items():
        if cond in condition.lower():
            rules.append(rule)
    return "\n".join(rules)

# ── Saved recipes ─────────────────────────────────────────────────────────────
def load_recipes():
    if os.path.exists(RECIPES_FILE):
        with open(RECIPES_FILE) as f:
            return json.load(f)
    return []

def save_recipes(data):
    with open(RECIPES_FILE, "w") as f:
        json.dump(data, f)

# ── Error handler ─────────────────────────────────────────────────────────────
@app.errorhandler(Exception)
def handle_exception(e):
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    try:
        data        = request.get_json(force=True, silent=True) or {}
        ingredients = data.get("ingredients", "")
        condition   = data.get("condition", "none")
        if not ingredients.strip():
            return jsonify({"error": "No ingredients provided"}), 400
        safe, removed = analyse_ingredients(ingredients, condition)
        return jsonify({"safe": safe, "removed": removed})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/dishes", methods=["POST"])
def api_dishes():
    try:
        if not groq_client:
            return jsonify({"error": "GROQ_KEY not configured"}), 500

        data        = request.get_json(force=True, silent=True) or {}
        safe_list   = data.get("safe_ingredients", [])
        condition   = data.get("condition", "")
        allergies   = data.get("allergies", "")
        constraints = data.get("constraints", "")
        safe_names  = ", ".join([i["name"] for i in safe_list]) or "general healthy ingredients"
        cond_rules  = get_condition_rules(condition)

        prompt = f"""You are a strict clinical AI nutritionist and chef.
USER HEALTH PROFILE: Medical: {condition or 'None'} | Allergies: {allergies or 'None'} | Constraints: {constraints or 'None'}
ML-VERIFIED SAFE INGREDIENTS: {safe_names}

{cond_rules}

CRITICAL RULES:
1. ONLY use ingredients from the ML-verified list above.
2. Follow the medical condition rules STRICTLY — they override everything.
3. Never suggest dishes that are medically harmful for the condition even if the ingredient passed ML check.
4. Provide 2-3 dietary tips specific to the condition.
5. Suggest EXACTLY 3 dish names appropriate for the condition.

FORMAT EXACTLY:
RECOMMENDATIONS:
- tip 1
- tip 2

DISHES:
1. Dish Name (Safe because: [specific medical reason])
2. Dish Name (Safe because: [specific medical reason])
3. Dish Name (Safe because: [specific medical reason])"""

        text = call_groq([{"role": "user", "content": prompt}])

        recommendations, dishes = "", []
        if "DISHES:" in text:
            parts = text.split("DISHES:")
            recommendations = parts[0].replace("RECOMMENDATIONS:", "").strip()
            for line in parts[1].split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    dishes.append(line)
        else:
            for line in text.split("\n"):
                line = line.strip()
                if line and line[0].isdigit():
                    dishes.append(line)

        return jsonify({"recommendations": recommendations, "dishes": dishes[:3]})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/recipe", methods=["POST"])
def api_recipe():
    try:
        if not groq_client:
            return jsonify({"error": "GROQ_KEY not configured"}), 500

        data        = request.get_json(force=True, silent=True) or {}
        dish_name   = data.get("dish_name", "")
        condition   = data.get("condition", "")
        allergies   = data.get("allergies", "")
        constraints = data.get("constraints", "")
        cond_rules  = get_condition_rules(condition)

        prompt = f"""Generate a DETAILED, SAFE recipe for "{dish_name}".
Medical condition: {condition or 'None'}
Allergies: {allergies or 'None'}
Constraints: {constraints or 'None'}

{cond_rules}

Include:
- Why this dish is safe and beneficial for the condition
- Full ingredients list with exact amounts
- Step-by-step cooking instructions (numbered)
- Estimated nutrition per serving (calories, protein, carbs, fat)
- One specific health tip for the condition

Be strict about the medical rules. If the dish name conflicts with the condition, adapt it to be safe."""

        recipe_text = call_groq([{"role": "user", "content": prompt}])
        return jsonify({"recipe": recipe_text, "dish": dish_name})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/recipes", methods=["GET"])
def get_recipes():
    return jsonify(load_recipes())

@app.route("/api/recipes", methods=["POST"])
def save_recipe():
    try:
        data    = request.get_json(force=True, silent=True) or {}
        recipes = load_recipes()
        recipes.append({
            "name":       data.get("name", "Unnamed"),
            "recipe":     data.get("recipe", ""),
            "condition":  data.get("condition", ""),
            "date_saved": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        })
        save_recipes(recipes)
        return jsonify({"ok": True, "count": len(recipes)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/recipes/<int:idx>", methods=["DELETE"])
def delete_recipe(idx):
    try:
        recipes = load_recipes()
        if 0 <= idx < len(recipes):
            recipes.pop(idx)
            save_recipes(recipes)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/image")
def proxy_image():
    import requests as req
    dish = request.args.get("q", "healthy food")
    if not UNSPLASH_KEY:
        return jsonify({"url": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=600&auto=format&fit=crop"})
    try:
        r   = req.get("https://api.unsplash.com/search/photos",
                      headers={"Authorization": f"Client-ID {UNSPLASH_KEY}"},
                      params={"query": f"{dish} food dish", "per_page": 1}, timeout=4)
        url = r.json().get("results", [{}])[0].get("urls", {}).get("regular", "")
    except:
        url = ""
    return jsonify({"url": url or "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=600&auto=format&fit=crop"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
