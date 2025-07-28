from flask import Flask, request, render_template
import joblib
import pandas as pd
import json
import numpy as np

app = Flask(__name__)

# Load artifacts
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')
with open('model_features.json') as f:
    feature_names = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form
        
        # Create input dictionary with all expected features
        input_data = {
            'age': float(form_data['age']),
            'avg_txn_amount': float(form_data['avg_txn_amount']),
            'txn_volatility': float(form_data['txn_volatility']),
            'txn_frequency': float(form_data['txn_frequency']),
            'three_months_trend': float(form_data['three_months_trend']),
            'duration_since_signup': float(form_data['duration_since_signup']),
            'source_of_income': form_data['source_of_income']
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # One-hot encoding
        input_df = pd.get_dummies(input_df, columns=['source_of_income'])
        
        # Ensure all expected columns exist
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Scale numerical features
        num_cols = ['age', 'avg_txn_amount', 'txn_volatility', 
                   'txn_frequency', 'three_months_trend', 'duration_since_signup']
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        result = 'Churned' if prediction == 1 else 'Not Churned'
        
        return render_template('result.html', prediction=result)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)