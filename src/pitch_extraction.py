import librosa
import numpy as np

def extract_pitch(audio_file):

    y, sr = librosa.load(audio_file)

    y = librosa.effects.preemphasis(y)
    y = librosa.util.normalize(y)
    y, _ = librosa.effects.trim(y)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_track = []

    for i in range(pitches.shape[1]):
        index = magnitudes[:, i].argmax()
        pitch = pitches[index, i]
        if pitch > 0:
            pitch_track.append(pitch)

    return np.array(pitch_track)

def normalize_pitch(pitch_track):
    
    pitch_track = pitch_track[pitch_track > 0]
    pitch_track = np.log1p(pitch_track)

    if len(pitch_track) == 0:
        return pitch_track
    
    pitch_track = np.convolve(pitch_track, np.ones(5)/5, mode='same')
    
    return pitch_track / np.mean(pitch_track)

def extract_chroma(audio_file):

    y, sr = librosa.load(audio_file)

    y = librosa.effects.preemphasis(y)
    y = librosa.util.normalize(y)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    return np.mean(chroma, axis=1)


def resize_melody(melody, target_length=100):

    if len(melody) == 0:
        return melody
    
    return np.interp(np.linspace(0, len(melody), target_length), np.arange(len(melody)), melody)
