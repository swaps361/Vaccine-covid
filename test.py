import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# New dataset
data = pd.DataFrame({
    'AGE': [32, 55, 28, 50, 60, 42, 48, 44, 41, 30, 49, 43, 21, 59, 63, 67],
    'GENDER': [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    'VACCINE_TYPE': [0, 2, 1, 0, 2, 0, 0, 0, 0, 1, 2, 1, 0, 1, 2, 1],
    'ADVERSE_REACTION': ['Fatigue', 'Headache', 'Fever', 'Fatigue', 'Fever', 
                         'Sore Arm', 'Chills', 'Fatigue', 'Sore Arm', 
                         'Headache', 'Chills', 'Fever', 'Fatigue', 
                         'Chills', 'Headache', 'Sore Arm']
})

# Encoding the target variable
le = LabelEncoder()
data['ADVERSE_REACTION'] = le.fit_transform(data['ADVERSE_REACTION'])

# Prepare features and target variable
X = data[['AGE', 'GENDER', 'VACCINE_TYPE']]
y = data['ADVERSE_REACTION']

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_scaled, y)

# Save model, scaler, and label encoder
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model, scaler, and label encoder saved successfully.")
