import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import json

# Load data
snapshot_df = pd.read_csv('customers.csv')

# Prepare features and target
y = snapshot_df['churn_label']
X = snapshot_df.drop(['customer_id', 'snapshot_date', 'churn_label'], axis=1)

# Preprocessing
X = pd.get_dummies(X, columns=['source_of_income'], drop_first=True)
X = X.fillna(X.mean())

# Save feature names for later use in Flask
feature_names = list(X.columns)
with open('model_features.json', 'w') as f:
    json.dump(feature_names, f)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['age', 'avg_txn_amount', 'txn_volatility', 
            'txn_frequency', 'three_months_trend', 'duration_since_signup']
X[num_cols] = scaler.fit_transform(X[num_cols])

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'churn_model.pkl')

print("Model training complete! Files saved:")
print("- churn_model.pkl (model)")
print("- scaler.pkl (scaler)")
print("- model_features.json (feature names)")