import librosa
import pandas as pd
import h5py
import numpy as np

timestamps =  [15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 
                20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 
                25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 
                30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 
                35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 
                40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500]

def load_and_clip_audio(file_path, clipNum):
    audio, sr = librosa.load(file_path, sr=44100)
    #print(file_path.split('/')[-1] + " - " + str(clipNum))

    # Based on clipnum find the starting and add 15 (15-25, 25-35, 35-45)
    #audio = audio[(((clipNum-1) * 10 * sr) + 15) : ((clipNum * 10 * sr) + 15)]
    clipnum = clipNum/1000
    audio = audio[int(clipNum/1000*sr) : int((clipNum+500)/1000*sr)]
    return audio

def compute_mel_spectrogram(file_path, audio, n_fft=2048, hop_length=512, n_mels=128):
    #_, sr = librosa.load(file_path, sr=None)
    sr =44100
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = np.log1p(mel_spectrogram)  # Apply logarithm to stabilize values

    return log_mel_spectrogram

def genFilepath(input):
    #input = input.split("_")
    return f"data/clips_45seconds_wav_resamp/{input}.wav"
    #return f"/content/drive/MyDrive/main/data/clips/{input[0]}.wav"



def genClip(input):
    input = input.split("_")
    return int(input[1])

def store_mfcc_data(mfcc_data, song_number, clip_number, file_path):
    with h5py.File(file_path, 'a') as hf:
        # Create a dataset for MFCC data
        name = f"mel_{song_number}_{clip_number}"
        hf.create_dataset(name, data=mfcc_data)

        # Add metadata as attributes
        hf[name].attrs['song_number'] = song_number
        hf[name].attrs['clip_number'] = clip_number

df = pd.read_csv('data/annotations/arousal_cont_average.csv')

df['FILEPATH'] = df['song_id'].apply(genFilepath)
#df['CLIP'] = df['song_id'].apply(genClip)

for row in df['song_id']:
    filepath = genFilepath(row)
    for timestamp in timestamps: 
        audio = load_and_clip_audio(filepath, timestamp)
        mfcc = compute_mel_spectrogram(filepath, audio)
        
        print(f"Song: {row}, Clip: {timestamp}, MFCC Shape: {mfcc.shape}")

        store_mfcc_data(mfcc, row, timestamp, "data/dataset.hdf5")

'''
with h5py.File('data/mel_10s.hdf5', 'r') as hf:
    # Iterate over all MFCC datasets and find the matching one
    for dataset_name in hf.keys():
        print(dataset_name)

while True:
    str = input("enter: ")
    with h5py.File('data/mel_10s.hdf5', 'r') as hf:
        # Iterate over all MFCC datasets and find the matching one
        for dataset_name in hf.keys():
            print(dataset_name)
            if f'mel_{str}' in dataset_name:
                dataset = hf[dataset_name]
                mfcc_data = dataset[:]
                print(mfcc_data)
                break
'''