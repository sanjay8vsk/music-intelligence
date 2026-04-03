import joblib
from ml_features import extract_features
import numpy as np

model = joblib.load("models/genre_model.pkl")

le = joblib.load("models/label_encoder.pkl")

scaler = joblib.load("models/scaler.pkl")

def predict_genre(file_path):
    features = extract_features(file_path)

    if features is None or np.isnan(features).any():
        return "Invalid audio"

    features = features.reshape(1, -1)
    features = scaler.transform(features)


    probs = model.predict_proba(features)
    pred = np.argmax(probs)

    genre = le.inverse_transform([pred])[0]
    confidence = np.max(probs)

    return genre, confidence
    
if __name__ == "__main__":
    file = "data/raw_audio/sample-12s.wav"
    
    genre, confidence = predict_genre(file)
    print(f"Predicted: {genre} ({confidence * 100:.2f}%)")

