import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your dataset (adjust path if needed)
df = pd.read_csv("synthetic_dataset_10k.csv")

# Separate features and target
X = df.drop('Health Condition', axis=1)
y = df['Health Condition']

# Normalize the temperature column
scaler = StandardScaler()
X[['Body temperature (°C)']] = scaler.fit_transform(X[['Body temperature (°C)']])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"✅ Accuracy: {accuracy * 100:.2f}%\n")
print("📊 Classification Report:\n", report)
print("🔁 Confusion Matrix:\n", conf_matrix)

print(df['Health Condition'].value_counts())

