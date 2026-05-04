# 🌿 CareCook — AI Healthy Recipe Generator

> Cook safely for your health. CareCook filters every ingredient through a dual-layer AI safety check, then generates personalized recipes using Llama-3.

🔗 **Live Demo:** https://carecook.onrender.com

## ✨ What It Does

CareCook is a medically-aware recipe generator built for people managing chronic health conditions. It verifies every ingredient through a machine learning model and condition-specific nutrient thresholds before a single dish is recommended.

## 🏥 Supported Conditions

🩸 Diabetes · 💊 PCOS/PCOD · ❤️ High Blood Pressure · 🫀 Heart Disease · 🫘 Kidney Disease · ⚖️ Obesity · 🟡 Jaundice · 🙂 None

## 🔬 How It Works

Each ingredient is matched against a database of 3,000+ food items. A **Random Forest ML model** scores it as safe, caution, or unsafe. Nutrient values are then checked against your condition's strict limits. Only ingredients that pass both checks reach **Llama-3 70B**, which generates dish suggestions and a full personalized recipe.

## 🛠️ Tech Stack

Python · Flask · scikit-learn (Random Forest) · Llama-3.3-70B via Groq API · Vanilla JS · Render

## 🚀 Run Locally

```bash
git clone https://github.com/Sumedha-15/CareCook.git
cd CareCook
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_KEY=your_groq_api_key
SECRET_KEY=any_random_string
UNSPLASH_KEY=your_unsplash_key
```

Train the model then run:
```bash
python train_model.py
python app.py
```
Visit `http://localhost:5000`

## 📁 Project Structure

```
CareCook/
├── app.py                    # Flask backend & API routes
├── model_logic.py            # ML inference + safety logic
├── train_model.py            # Model training script
├── condition_thresholds.json # Per-condition nutrient limits
├── requirements.txt
├── Procfile
├── templates/index.html
└── static/css/ & js/
```

## 🔐 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_KEY` | Groq API key for Llama-3 |
| `SECRET_KEY` | Flask session secret |
| `UNSPLASH_KEY` | Unsplash key for dish images (optional) |

## 👥 Team

Built by **Sumedha Modi and Aditi Gupta ** as a final AI project.

## 📄 License

MIT License
