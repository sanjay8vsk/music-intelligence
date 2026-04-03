import librosa

import numpy as np

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)

        y = librosa.util.normalize(y)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        delta = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta, axis=1)
        delta_std = np.std(delta, axis=1)
        # chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)
        contrast_std = np.std(spectral_contrast, axis=1)
        
        
        features = np.hstack([mfcc_mean, mfcc_std, delta_mean, delta_std, chroma_mean, chroma_std, contrast_mean, contrast_std])

        return features
    except Exception as e:
        print("Feature extraction failed:", file_path)
        print("Reason:", e)
        return None