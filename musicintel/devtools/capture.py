import sounddevice as sd
from scipy.io.wavfile import write
import time

def record_audio(filename="query.wav", duration=5, sample_rate=22050):
    print("Recording starting in 3 seconds...")

    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)

    sd.wait()

    write(filename, sample_rate, recording)

    print("Recording saved as", filename)

    return filename 

if __name__ == "__main__":
    record_audio()