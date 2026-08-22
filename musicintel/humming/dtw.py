import librosa 

def compare_melodies(query, target):

    distance = librosa.sequence.dtw(query.reshape(1, -1), target.reshape(1, -1), metric="euclidean")[0][-1, -1]

    normalized = distance / len(query)
    
    return normalized