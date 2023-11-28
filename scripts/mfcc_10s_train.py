import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
#import torch_xla
#import torch_xla.core.xla_model as xm

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import os

#device = xm.xla_device()

class NoM1GPUAcc(Exception): 
    pass


if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)
    del x
else:
    print ("MPS device not found.")
    raise NoM1GPUAcc()

def load_and_clip_audio(file_path, clipNum):
    audio, sr = librosa.load(file_path, sr=None)
    print(file_path.split('/')[-1] + " - " + str(clipNum))

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
        mfcc = compute_mfcc(file_path, audio)
        print(mfcc.shape)
        return mfcc, torch.tensor(real_output, dtype=torch.float32)

class MelRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MelRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, output_size[0])
        self.fc2 = nn.Linear(hidden_size, output_size[0])
        self.fc3 = nn.Linear(hidden_size, output_size[1])

    def forward(self, x):
        # Input x should have shape (batch_size, sequence_length, input_size)
        output, hidden = self.rnn(x)

        # Take the output from the last time step
        last_output = output[:, -1, :]

        # Apply fully connected layers for each output
        output1 = self.fc1(last_output)
        output2 = self.fc2(last_output)
        output3 = self.fc3(last_output)

        return output1, output2, output3

def genFilepath(input):
    input = input.split("_")
    return f"data/clips_45seconds_wav_resamp/{input[0]}.wav"
    #return f"/content/drive/MyDrive/main/data/clips/{input[0]}.wav"

def genClip(input):
    input = input.split("_")
    return int(input[1])

df = pd.read_csv('data/annotations_new/combined_10.csv')
#df = pd.read_csv('/content/drive/MyDrive/main/data/annotations_new/combined_10.csv')
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
hidden_size = 16  # adjust as needed based on *complexity*
output_size = (16, 2)  # adjust as needed

model = MelRNN(input_size, hidden_size, output_size)
model = model.to(device)

# Use Mean Squared Error (MSE) loss for regression task
criterion = nn.MSELoss()

# Use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint_path = 'models/10second_temp.pth'

if os.path.exists(checkpoint_path):
    # Load the checkpoint to resume training
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
    losses = checkpoint['losses']
    print(f"Resuming training from epoch {start_epoch}")

else:
    # If no checkpoint, start training from the beginning
    start_epoch = 0


batch_size = 128
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle the training set
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle the test set

# Training loop
num_epochs = 1
losses = []

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(start_epoch, num_epochs):
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []

    # Training
    model.train()
    for batch in train_dataloader:
        input_data, target_coordinates = batch

        input_data = input_data.to(device)
        target_coordinates = target_coordinates.to(device)

        # Send data to TPU device
        #input_data = xm.send(input_data, device=device)
        #target_coordinates = xm.send(target_coordinates, device=device)

        output_coordinates = model(input_data)

        # Compute the loss
        loss = criterion(output_coordinates[2], target_coordinates)

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

            input_data = input_data.to(device)
            target_coordinates = target_coordinates.to(device)

            output_coordinates = model(input_data)

            # Compute the loss
            val_loss = criterion(output_coordinates[2], target_coordinates)
            epoch_val_losses.append(val_loss.item())

            # Save predictions and targets for Precision-Recall curve
            predictions_list.append(output_coordinates[1].cpu().numpy())
            targets_list.append(target_coordinates.cpu().numpy())

            predictions = output_coordinates[1].argmax(dim=1)  # Assuming classification task
            accuracy = accuracy_score(target_coordinates.argmax(dim=1).cpu().numpy(), predictions.cpu().numpy())
            f1 = f1_score(target_coordinates.argmax(dim=1).cpu().numpy(), predictions.cpu().numpy(), average='weighted')

            epoch_val_accuracies.append(accuracy)

            print(f'Batch Accuracy: {accuracy:.4f}, Batch F1 Score: {f1:.4f}')

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

    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': losses,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f'Model checkpoint saved at epoch {epoch + 1}')




# Save the trained model

torch.save(model.state_dict(), 'models/10second_final.pth')



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