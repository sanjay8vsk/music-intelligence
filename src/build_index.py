import os
import numpy as np
import faiss

FEATURE_FOLDER = "data/features"

def build_faiss_index():

    embeddings = []
    song_names = []

    for file in os.listdir(FEATURE_FOLDER):

        if file.endswith(".npy"):

            path = os.path.join(FEATURE_FOLDER, file)
            mfcc = np.load(path)
            
            mean = np.mean(mfcc, axis=1)
            std = np.std(mfcc, axis=1)
            embedding = np.concatenate([mean, std])

            embeddings.append(embedding)
            song_names.append(file)
    embeddings = np.vstack(embeddings).astype("float32")

    embeddings = np.ascontiguousarray(embeddings).astype("float32")

    print("Embeddings shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    
    index.add(embeddings) # type: ignore

    return index, song_names

if __name__ == "__main__":
    index, songs = build_faiss_index()

    print("songs indexed:", len(songs))