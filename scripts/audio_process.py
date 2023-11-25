import torchaudio
import torch
import os

directory = 'data/clips_45seconds_wav'
 
# iterate over files in
# that directory
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        waveform, sample_rate = torchaudio.load(f)
        # Frequency (sample rate) of the audio
        print(f"Sample Rate: {sample_rate} Hz")

        # Length of the audio clip in seconds
        audio_length_seconds = waveform.shape[1] / sample_rate
        print(f"Audio Length: {audio_length_seconds} seconds")

        # If you want the length in samples, you can use the shape of the waveform directly
        audio_length_samples = waveform.shape[1]
        print(f"Audio Length (samples): {audio_length_samples} samples")