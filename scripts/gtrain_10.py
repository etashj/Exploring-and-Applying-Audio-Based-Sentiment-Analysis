import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets

import torchaudio
import torchaudio.transforms as T
from torchvision import transforms

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

def load_and_clip_audio(file_path, clipNum):
    audio, sr = librosa.load(file_path, sr=None)
    print(file_path.split('/')[-1] + " - " + str(clipNum))

    audio = audio[((clipNum-1) * 10 * sr):(clipNum * 10 * sr)]
    return audio

def compute_mel_spectrogram(file_path, audio, n_fft=2048, hop_length=512, n_mels=13):
    #_, sr = librosa.load(file_path, sr=None)
    sr =44100
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    log_mel_spectrogram = np.log1p(mel_spectrogram)  # Apply logarithm to stabilize values

    return log_mel_spectrogram

class MyDataset(Dataset):
    def __init__(self, file_paths, clips, real_outputs):
        self.file_paths = file_paths
        self.clips = clips
        self.real_outputs = real_outputs

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        clip = self.clips[idx]
        real_output = self.real_outputs[idx]

        # Load and process audio
        audio = load_and_clip_audio(file_path, clip)
        mel_spectrogram = compute_mel_spectrogram(file_path, audio)
        #print(mel_spectrogram.shape)
        return mel_spectrogram, torch.tensor(real_output, dtype=torch.float32)

class MelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MelRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size[0])
        self.fc2 = nn.Linear(hidden_size, output_size[1])

    def forward(self, x):
        # Input x should have shape (batch_size, sequence_length, input_size)
        output, hidden = self.rnn(x)
        
        # Take the output from the last time step
        last_output = output[:, -1, :]

        # Apply fully connected layers for each output
        output1 = self.fc1(last_output)
        output2 = self.fc2(last_output)

        return output1, output2

def genFilepath(input):
    input = input.split("_")
    return f"data/clips_45seconds_wav_resamp/{input[0]}.wav"

def genClip(input):
    input = input.split("_")
    return int(input[1])

df = pd.read_csv('data/annotations_new/combined_10.csv')
df['FILEPATH'] = df['SONG_ID'].apply(genFilepath)
df['CLIP'] = df['SONG_ID'].apply(genClip)

real_outputs = df[['AROUSAL_AVG', 'VALENCE_AVG']].values

dataset = MyDataset(df['FILEPATH'].tolist(), df['CLIP'].tolist(), real_outputs)

# Example usage:
input_size = 862  # replace with your actual input size, number of bins in the mel spectrogram
hidden_size = 64  # adjust as needed based on *complexity*
output_size = (64, 2)  # adjust as needed

model = MelRNN(input_size, hidden_size, output_size)

# Use Mean Squared Error (MSE) loss for regression task
criterion = nn.MSELoss()

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

'''
# Generate random input data and target coordinates (replace with your actual data)
batch_size = 16  # number of training examples utilized in one iteration: more = faster but more memory
sequence_length = 3 # Sequence length is the number of time steps in your input data (i.e. 10s = 3 steps for 30s)
input_data = torch.rand((batch_size, sequence_length, input_size))
target_coordinates = torch.rand((batch_size, 2))
'''
batch_size = 50
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 2

for epoch in range(num_epochs):
    for batch in dataloader:
        input_data, target_coordinates = batch

        # Forward pass
        output_coordinates = model(input_data)
        print(output_coordinates[1].shape)
        # Compute the loss
        loss = criterion(output_coordinates[1], target_coordinates)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss for monitoring during training
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Save the trained model
torch.save(model.state_dict(), 'mel_rnn_model.pth')










