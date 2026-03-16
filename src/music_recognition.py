import librosa
import numpy as np
from build_index import build_faiss_index

def recognize_song(audio_file):

    y, sr = librosa.load(audio_file)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    query_embedding = np.mean(mfcc, axis=1).astype("float32").reshape(1, -1)

    index, song_names = build_faiss_index()

    distances, indices = index.search(query_embedding, k=3) # type: ignore

    results = []

    for i in indices[0]:
        results.append(song_names[i])
    return results

if __name__ == "__main__":

    query = "data/raw_audio/sample-9s.wav"

    result = recognize_song(query)

    print("Top matches:")
    for song in result:
        print(song)