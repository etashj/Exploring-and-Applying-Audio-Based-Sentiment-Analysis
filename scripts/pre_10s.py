import librosa
import pandas as pd
import h5py

def load_and_clip_audio(file_path, clipNum):
    audio, sr = librosa.load(file_path, sr=None)
    #print(file_path.split('/')[-1] + " - " + str(clipNum))

    # Based on clipnum find the starting and add 15 (15-25, 25-35, 35-45)
    audio = audio[(((clipNum-1) * 10 * sr) + 15) : ((clipNum * 10 * sr) + 15)]
    return audio

def compute_mfcc(file_path, audio, n_fft=2048, hop_length=512, n_mfcc=13):
    # _, sr = librosa.load(file_path, sr=None)
    sr = 44100
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc
    )
    return mfcc

def genFilepath(input):
    input = input.split("_")
    return f"data/clips_45seconds_wav_resamp/{input[0]}.wav"
    #return f"/content/drive/MyDrive/main/data/clips/{input[0]}.wav"

def genNum(input):
    input = input.split("_")
    return int(input[0])

def genClip(input):
    input = input.split("_")
    return int(input[1])

def store_mfcc_data(mfcc_data, song_number, clip_number, file_path):
    with h5py.File(file_path, 'a') as hf:
        # Create a dataset for MFCC data
        name = f"mfcc_{song_number}_{clip_number}"
        hf.create_dataset(name, data=mfcc_data)

        # Add metadata as attributes
        hf[name].attrs['song_number'] = song_number
        hf[name].attrs['clip_number'] = clip_number

df = pd.read_csv('data/annotations_new/combined_10.csv')

df['FILEPATH'] = df['SONG_ID'].apply(genFilepath)
df['CLIP'] = df['SONG_ID'].apply(genClip)

for row in df['SONG_ID']:
    filepath = genFilepath(row)
    clip = genClip(row)

    audio = load_and_clip_audio(filepath, clip)
    mfcc = compute_mfcc(filepath, audio)
    
    print(f"Song: {genNum(row)}, Clip: {clip}, MFCC Shape: {mfcc.shape}")

    store_mfcc_data(mfcc, genNum(row), clip, "data/mfcc_10s.hdf5")


with h5py.File('data/mfcc_10s.hdf5', 'r') as hf:
    # Iterate over all MFCC datasets and find the matching one
    for dataset_name in hf.keys():
        print(dataset_name)

while True:
    str = input("enter: ")
    with h5py.File('data/mfcc_10s.hdf5', 'r') as hf:
        # Iterate over all MFCC datasets and find the matching one
        for dataset_name in hf.keys():
            print(dataset_name)
            if f'mfcc_{str}' in dataset_name:
                dataset = hf[dataset_name]
                mfcc_data = dataset[:]
                print(mfcc_data)