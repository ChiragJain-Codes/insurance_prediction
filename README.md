
# 📄 Insurance Premium Prediction

## Overview
A machine learning model to predict individual medical insurance costs (or premiums) based on personal and demographic information (e.g., age, sex, BMI, smoking status, region).

## 🧠 Motivation
Understanding how different factors influence insurance cost can help users anticipate healthcare expenses and enable insurance companies to personalize pricing.

## 📦 Dataset
- The project uses a dataset similar to the Kaggle “Medical Cost Personal Datasets” (with features such as `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`).  
- **Source**: [Link to Kaggle or original dataset]

## 🛠 Installation & Setup
```bash
git clone https://github.com/ChiragJain-Codes/insurance_prediction.git
cd insurance_prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🧪 Usage
- **Jupyter notebook** (`insurance_prediction.ipynb` or equivalent):
  - Contains exploratory data analysis, data preprocessing, model training, hyperparameter tuning (e.g., linear regression, ridge regressor, random forest, XGBoost), and evaluation.
- **Web app** (`app.py`):
  - A simple Flask (or Streamlit) application allowing users to input personal information (age, BMI, etc.) and get a predicted insurance premium.
- **Model artifact**:
  - Trained model saved as `model.pkl` (or similar).

## 🔧 Project Workflow / Features
1. Data exploration and visualization  
2. Encoding categorical variables & feature engineering  
3. Splitting into train and test sets  
4. Model training and comparison (e.g., Linear Regression, Random Forest, XGBoost)  
5. Hyperparameter tuning (e.g., grid search, cross-validation)  
6. Model evaluation using RMSE, R², etc.  
7. Saving the trained model for deployment  
8. Creating a user-facing interface for predictions  

## 📈 Results
- Example: “Random Forest Regressor achieved the best performance with ~86% accuracy or \(R^2 pprox 0.86\)”  
- Include result tables or visualizations (e.g., model comparison plots, residual plots).

## 🚀 Deployment
- Web app deployed on Heroku / Streamlit Cloud / AWS etc.  
- **Demo link**: `https://your-app-url`  
- Instructions for local testing and usage included.

## 📚 Tech Stack
- Python 3.x  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib` / `seaborn`, (`xgboost`), `flask` (or `streamlit`)  
- Model Serialization: `pickle` (or `joblib`)  
- Deployment: Heroku / Streamlit Cloud / Docker (optional)

## 💼 Usage Example (Flask)
```python
import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
input_data = pd.DataFrame({
    'age': [30],
    'sex': ['male'],
    'bmi': [28.5],
    'children': [1],
    'smoker': ['no'],
    'region': ['southwest']
})
prediction = model.predict(input_data)
print(f'Predicted insurance cost: ₹{prediction[0]:.2f}')
```

## 🚧 Contributing
1. Fork the repo  
2. Create a feature branch  
3. Make your changes & commit  
4. Submit a pull request

## 📝 License
This project is licensed under the MIT License.

## 📜 Acknowledgments
- Credit to the original dataset (e.g., Kaggle “Medical Cost Personal Datasets”)  
- Inspired by various tutorials and blog posts on insurance cost prediction using ML  
- [Optional] Deployed via Flask / Streamlit / Heroku

## 🎯 Future Work
- Add deeper feature engineering (e.g., polynomial features)  
- Integrate user interface improvements  
- Explore deep learning models (e.g., Neural Networks)  
- Add real-time API support / Docker containerization  
- Expand dataset with more features (e.g., pre-existing conditions)


## 📁 Project Structure

```
insurance_prediction/
│
├── data/                     # Raw and processed datasets
│   └── insurance.csv
│
├── notebooks/                # Jupyter notebooks for EDA and modeling
│   └── insurance_prediction.ipynb
│
├── models/                   # Saved models (pickle/joblib files)
│   └── model.pkl
│
├── app/                      # Web application
│   ├── app.py                # Flask or Streamlit app
│   ├── templates/            # HTML templates (Flask)
│   │   └── index.html
│   └── static/               # Static files like CSS, JS
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview (this file)
└── .gitignore                # Git ignore file
```
