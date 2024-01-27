import librosa
import librosa.display
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})
plt.style.use('seaborn-v0_8')

def cut_audio(y, sr, start_time, end_time):

    # Calculate start and end samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Plot the waveform of the cut segment
    plt.figure(figsize=(14, 5), dpi=300)
    plt.xlim(start_time, end_time)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Cut Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig("latex/images/audio/waveform10.png")
    plt.show()

import numpy as np

def run_fft(y, sr):
    # Compute the short-time Fourier transform (STFT)
    D = librosa.stft(y)

    # Plot the magnitude spectrogram
    plt.figure(figsize=(14, 5), dpi=300)
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time', sr=sr)
    plt.title('FFT Magnitude Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig("latex/images/audio/FFT10.png")
    plt.show()

def mel_spectrogram(y, sr):
    # Compute the mel spectrogram
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128
    )

    # Plot the mel spectrogram
    plt.figure(figsize=(14, 5), dpi=300)
    librosa.display.specshow(librosa.power_to_db(mel_spec, ref=np.max), y_axis='mel', x_axis='time', sr=sr)
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.savefig("latex/images/audio/MelSpectrogram10.png")
    plt.show()


def mfcc(y, sr):
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Plot the MFCCs
    plt.figure(figsize=(14, 5), dpi=300)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCCs')
    plt.savefig("latex/images/audio/MFCC10.png")
    plt.show()

if __name__ == "__main__":
    mp3_file_path = 'data/clips_45seconds_wav_resamp/10.wav'
    start_time = 15  # in seconds
    end_time = 45    # in seconds

    y, sr = librosa.load(mp3_file_path, sr=None)
    cut_audio(y, sr, start_time, end_time)

    run_fft(y, sr)
    mel_spectrogram(y, sr)
    mfcc(y, sr)
