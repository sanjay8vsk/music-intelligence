import os
import numpy as np
from pitch_extraction import extract_pitch, normalize_pitch
from pitch_extraction import resize_melody, extract_chroma

AUDIO_FOLDER = "data/raw_audio"

def build_melody_database():

    melody_db = {}

    for file in os.listdir(AUDIO_FOLDER):
        if file.endswith((".wav", ".mp3", ".flac")):
            path = os.path.join(AUDIO_FOLDER, file)

            pitch = extract_pitch(path)
            melody = normalize_pitch(pitch)

            chroma = extract_chroma(path)

            combined = np.concatenate([melody, chroma])

            if len(combined) > 0:
                melody_db[file] = combined
    return melody_db

print("Total songs in database:", len(build_melody_database()))