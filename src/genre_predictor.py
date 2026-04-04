import joblib
from ml_features import extract_features
import numpy as np
import librosa
import soundfile as sf

model = joblib.load("models/genre_model.pkl")

le = joblib.load("models/label_encoder.pkl")

scaler = joblib.load("models/scaler.pkl")

def predict_genre(file_path):
    import librosa

    y, sr = librosa.load(file_path, duration=30)

    predictions = []
    confidences = []

    # 🔥 Split into 3-second chunks
    for i in range(0, len(y), int(sr * 8)):
        segment = y[i:i + int(sr * 8)]

        if len(segment) < sr:
            continue
        features = extract_features(y=segment, sr=sr)

        if features is None or np.isnan(features).any():
            continue

        

        if isinstance(features, float) or np.isnan(features).any():
            continue

        features = features.reshape(1, -1)
        features = scaler.transform(features)

        probs = model.predict_proba(features)[0]
        if np.max(probs) < 0.5:
            continue
        pred = np.argmax(probs)

        predictions.append(pred)
        confidences.append(np.max(probs))
    
    if len(predictions) == 0:
        return "No valid audio", 0.0
    
    print("Segments used:", len(confidences))
    print("Max confidence:", max(confidences)
    if confidences else 0)
    

    # 🔥 Majority vote
    from collections import defaultdict

    if len(predictions) == 0:
        return "No valid audio", 0.0

    vote_scores = defaultdict(float)
    for p, c in zip(predictions, confidences):
        vote_scores[p] += c
    if not vote_scores:
        return "Unknown", 0.0
    final_pred = max(vote_scores.items(), key=lambda x: x[1])[0]
    

    final_conf = (max(confidences) + (sum(confidences) / len(confidences))) / 2

    genre = le.inverse_transform([final_pred])[0]

    return genre, final_conf
if __name__ == "__main__":
    file = "data/raw_audio/sample-12s.wav"
    genre, confidence = predict_genre(file)
    print(f"Predicted: {genre} ({confidence * 100:.2f}%)")