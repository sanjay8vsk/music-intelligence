import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import joblib
from musicintel.analysis.features import extract_features
from sklearn.ensemble import GradientBoostingClassifier

DATASET_PATH = "data/ml_dataset"

X = []
y = []

for genre in os.listdir(DATASET_PATH):
    genre_path = os.path.join(DATASET_PATH, genre)

    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        print("Processing:", file_path)
        
        try:
            features = extract_features(file_path)

            if features is not None and not np.isnan(features).any():
                X.append(features)
                y.append(genre)
        except:
            print("Error processing file:", file_path)
print("Total samples:", len(X))
print("Feature shape:", np.array(X).shape)
print("Finished feature extraction")
X = np.array(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

print("Training accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))

joblib.dump(model, "models/genre_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model trained successfully.")