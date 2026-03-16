import numpy as np

def generate_embedding(mfcc):
    """
    Convert MFCC features into a fixed-size embedding vector
    """
    embedding = np.mean(mfcc, axis=1)
    return embedding