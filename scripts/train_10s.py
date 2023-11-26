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

from sklearn.model_selection import train_test_split

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

# Split the dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)  # Adjust test_size as needed

# Create datasets for training and test sets
train_dataset = MyDataset(train_df['FILEPATH'].tolist(), train_df['CLIP'].tolist(), train_df[['AROUSAL_AVG', 'VALENCE_AVG']].values)
test_dataset = MyDataset(test_df['FILEPATH'].tolist(), test_df['CLIP'].tolist(), test_df[['AROUSAL_AVG', 'VALENCE_AVG']].values)


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
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle the training set
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle the test set

# Training loop
num_epochs = 1
losses = []

num_epochs = 5
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = [] 

for epoch in range(num_epochs):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = [] 

    # Training
    model.train()
    for batch in train_dataloader:
        input_data, target_coordinates = batch
        output_coordinates = model(input_data)

        # Compute the loss
        loss = criterion(output_coordinates[1], target_coordinates)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_losses.append(loss.item())

        predictions = output_coordinates[1].argmax(dim=1)  # Assuming classification task
        accuracy = (predictions == target_coordinates.argmax(dim=1)).float().mean().item()
        epoch_train_accuracies.append(accuracy)

    # Validation
    model.eval()
    with torch.no_grad():
        predictions_list = []
        targets_list = []
        for batch in test_dataloader:
            input_data, target_coordinates = batch
            output_coordinates = model(input_data)

            # Compute the loss
            val_loss = criterion(output_coordinates[1], target_coordinates)
            epoch_val_losses.append(val_loss.item())

            # Save predictions and targets for Precision-Recall curve
            predictions_list.append(output_coordinates[1].cpu().numpy())
            targets_list.append(target_coordinates.cpu().numpy())

            predictions = output_coordinates[1].argmax(dim=1)  # Assuming classification task
            accuracy = (predictions == target_coordinates.argmax(dim=1)).float().mean().item()
            epoch_val_accuracies.append(accuracy)
            

    avg_train_loss = np.mean(epoch_train_losses)
    avg_val_loss = np.mean(epoch_val_losses)
    avg_train_accuracy = np.mean(epoch_train_accuracies)
    avg_val_accuracy = np.mean(epoch_val_accuracies)  
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(avg_train_accuracy)
    val_accuracies.append(avg_val_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
          f'Train Accuracy: {avg_train_accuracy:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')




# Save the trained model
torch.save(model.state_dict(), 'models/10second.pth')



plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()

plt.savefig('results/10second.png')

plt.show()

