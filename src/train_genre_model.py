import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from ml_features import extract_features

DATASET_PATH = "data/ml_dataset"

X = []
y = []

for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)

    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        
        try:
            features = extract_features(file_path)

            if features is not None and not np.isnan(features).any():
                X.append(features)
                y.append(genre)
        except:
            print("Error processing file:", file_path)
print("Total samples:", len(X))
print("Feature shape:", np.array(X).shape)
X = np.array(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

model = RandomForestClassifier(n_estimators=300, max_depth=20)
model.fit(X_train, y_train)

joblib.dump(model, "models/genre_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model trained successfully.")