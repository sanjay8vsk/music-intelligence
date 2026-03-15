import os
import librosa
import numpy as np

# assigning paths
AUDIO_FOLDER = "data/raw_audio"
FEATURE_FOLDER = "data/features"

def extract_mfcc(audio_path):
    """
    Extract MFCC features from an audio file
    """
    y, sr = librosa.load(audio_path, sr=22050, duration=30)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    return mfcc 

def process_audio_files():
    """
    Process all audio files in the AUDIO_FOLDER and save their MFCC features
    """

    os.makedirs(FEATURE_FOLDER, exist_ok=True)

    for file in os.listdir(AUDIO_FOLDER):

        if file.endswith((".wav", ".mp3", ".flac")):

            audio_path = os.path.join(AUDIO_FOLDER, file)
            print(f"\nLoading file: {audio_path}")

            mfcc = extract_mfcc(audio_path)

            feature_file = file.split(".")[0] + ".npy"
            save_path = os.path.join(FEATURE_FOLDER, feature_file)

            np.save(save_path, mfcc)

            print(f"Saved features -> {save_path}")

if __name__ == "__main__":
    process_audio_files()
