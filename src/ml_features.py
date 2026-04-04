import librosa

import numpy as np

def extract_features(file_path=None, y=None, sr=None):
    try:
        if file_path is not None:
            y, sr = librosa.load(file_path, duration=30)
        elif y is None or sr is None:
            raise ValueError("Either file_path or both y and sr must be provided.")

        y = librosa.util.normalize(y)

        y, _ = librosa.effects.trim(y)
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

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(spectral_contrast, axis=1)
        contrast_std = np.std(spectral_contrast, axis=1)

        # centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)
        
        
        features = np.hstack([mfcc_mean, mfcc_std, delta_mean, delta_std, chroma_mean, chroma_std, contrast_mean, contrast_std, zcr_mean, zcr_std, rms_mean, rms_std, centroid_mean, centroid_std])

        return features
    except Exception as e:
        print("Feature extraction failed:", file_path)
        print("Reason:", e)
        return None