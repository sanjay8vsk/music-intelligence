from record_audio import record_audio
from pitch_extraction import extract_pitch, normalize_pitch
from melody_match import compare_melodies
from build_melody_db import build_melody_database
from pitch_extraction import resize_melody
import numpy as np

def recognize_humming():

    query_file = record_audio()

    query_pitch = extract_pitch(query_file)
    
    
    from pitch_extraction import extract_chroma
    query_melody = normalize_pitch(query_pitch)
    query_melody = resize_melody(query_melody)
    query_chroma = extract_chroma(query_file)
    query_combined = np.concatenate([query_melody, query_chroma])

    db = build_melody_database()

    results = []

    for song, melody in db.items():

        melody = resize_melody(melody)

        score = compare_melodies(query_combined, melody)

        results.append((song, score))

        results.sort(key=lambda x: x[1])

        best_score = results[0][1]
        if best_score > 5:
            return []

        return results[:3]

if __name__ == "__main__":
    result = recognize_humming()
    
    print("\n🎵 Top matches:")

    if not result:
        print(" No confident matches found. Try humming more clearly or for a longer duration.")
    else:
        for i, (song, score) in enumerate(result, 1):
            confidence = np.exp(-score)
            print(f"{i}. {song} -> score: {score:.2f} | confidence: {confidence*100:.2f}%")