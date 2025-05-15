import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv('data/synthetic_cattle_data.csv')

# Prepare features and target
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Initialize and train model
model = XGBClassifier(
    objective='multi:softprob',
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Save model
with open('models/cattle_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully.")