from flask import Flask, render_template, request
import os
import pandas as pd
import joblib
import json
from datetime import datetime

app = Flask(__name__, template_folder='templates')

# Load the model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'insurance_model.joblib')
model = joblib.load(MODEL_PATH)

# File to store prediction history
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'prediction_history.json')

def load_history():
    """Load prediction history from file"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_prediction(input_data, prediction):
    """Save a new prediction to history"""
    history = load_history()
    
    # Add timestamp
    prediction_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'age': input_data['age'],
        'sex': input_data['sex'],
        'bmi': input_data['bmi'],
        'children': input_data['children'],
        'smoker': input_data['smoker'],
        'region': input_data['region'],
        'predicted_charges': prediction
    }
    
    history.append(prediction_record)
    
    # Save to file
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/project', methods=['GET', 'POST'])
def project():
    prediction = None
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form['sex']
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = request.form['smoker']
        region = request.form['region']

        # Encode categorical variables
        sex_dict = {'female': 0, 'male': 1}
        smoker_dict = {'no': 0, 'yes': 1}
        region_dict = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}

        # Prepare input for model
        input_df = pd.DataFrame([{
            'age': age,
            'sex': sex_dict[sex],
            'bmi': bmi,
            'children': children,
            'smoker': smoker_dict[smoker],
            'region': region_dict[region]
        }])

        prediction = model.predict(input_df)[0]
        prediction = round(prediction, 2)
        
        # Save prediction to history
        input_data = {
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region
        }
        save_prediction(input_data, prediction)

    return render_template('project.html', prediction=prediction)

@app.route('/history')
def history():
    historical_data = load_history()
    return render_template('history.html', historical_data=historical_data)

if __name__ == '__main__':
    app.run(debug=True)
