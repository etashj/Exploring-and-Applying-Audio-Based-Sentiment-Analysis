## Train the emotion predicition model for 0.5 second mel spectrograms (shape: 128, 44)

# Import neccesary packages
import random
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from models import MusicEmotionLSTM

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set seeds for reproducibility
random.seed(3407)
torch.manual_seed(3407)
np.random.seed(3407)

# Enable Metal acceleration for M1 GPUs if possible
if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)
    del x
else:
    print ("MPS device not found.")
    device = torch.device("cpu")

# Define a dataset class that takes mel spectrograms in a dictionary and dataframes for arousal and valence
class MelSpectrogramDataset(Dataset):
    def __init__(self, data, a_labels, v_labels, transform=None):
        self.data = data                # Mel Spectrogram in a pytorch tensor
        self.arousal_labels = a_labels  # Regression output
        self.valence_labels = v_labels  # Regression output
        self.transform = transform

    def __len__(self):
        return len(self.data)

    # MelSpectrogramDataset[id, timestamp]
    # Timestamp goes from 15000 to 43500 in increments of 500
    '''
     [15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 
     20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 
     25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 
     30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 
     35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 
     40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500]
    '''
    def __getitem__(self, index):
        ## index MUST be a tuple (id, timestamp)

        ## Z Score normalization on input data
        orig_data = self.data[str(index[0])+"_"+str(index[1])]
        normed_data = (orig_data-orig_data.mean())/orig_data.std()

        # Assemble dictionary for reutrn data
        sample = {'mel_data': normed_data, 
                  'arousal': self.arousal_labels.loc[self.arousal_labels['song_id'] == index[0]]["sample_"+str(index[1])+"ms"].values[0], 
                  'valence': self.valence_labels.loc[self.valence_labels['song_id'] == index[0]]["sample_"+str(index[1])+"ms"].values[0]}

        # Apply trasnform if applicable
        if self.transform:
            sample = self.transform(sample)

        return sample

# Custom sampler to call custom __getitem__ on MelSpectrogramDataset since it takes tuple indices
class SongSampler(Sampler):
    # Get every song and every timestep included as part of the dataset
    def __init__(self, dataset):
        l = [int(s.split("_")[0]) for s in list(dataset.data.keys())]
        self.songs = []
        [self.songs.append(x) for x in l if x not in self.songs]
        del l
        self.timestamps =  [15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 
                            20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 
                            25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 
                            30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 
                            35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 
                            40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500]

    # Create every posible combination of song and timestamp and reutnr an iterable (in order)
    def __iter__(self):
        ret = []
        random.shuffle(self.songs)
        for item in self.songs: 
            for time in self.timestamps: 
                ret.append([item, time])
        return iter(ret)
    
    def __len__(self):
        return len(self.songs)


## SPLIT: 
##     TRAIN: ~80%    596 songs    34568 samples
##     VAL  : ~10%    74 songs     4292 samples
##     TEST : ~10%    74 songs     4292 samples
## ----------------------------------------------
##     TOTAL: 100%    744 songs    43152 samples

timestamps =  [15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 
            20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 
            25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 
            30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 
            35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 
            40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500]

melData = {}  # Dictionary to be passed to Dataset after reading from hdf5 file with mel spectrograms
allSongIds = [] # len = 744
# Dictionaries for custom splits
melDataVal = {} 
melDataTest = {}

# Read from hdf5 file and assemble the melData dictionary, applying random sound to improve robustness
with h5py.File('data/dataset.hdf5', 'r') as file:
    print(file)
    for item in file:
        ## np.array(file[item]).shape    -->   (128, 173)
        #print(item, torch.tensor(np.array(file[item]).shape))

        noise_level = 0.5
        mel_spectrogram_noisy = np.array(file[item]) + noise_level * np.random.normal(0, 1, np.array(file[item]).shape)

        #melData[item] = torch.from_numpy(np.array(file[item])).to(device)
        melData[item[4:]] = torch.transpose(torch.from_numpy(mel_spectrogram_noisy).float().to(device),  0, 1)

        if item.split("_")[1] not in allSongIds: allSongIds.append(item.split("_")[1])

# Sample keys randomly to create validation and test set keys for songs
valKeys = random.sample(allSongIds, 74+74)
testKeys = random.sample(valKeys, 74)

# Remove duplicates between testkeys and valKeys
for key in testKeys: 
    if key in valKeys: 
        valKeys.remove(key)

# Get every timestap of every sampled validation song and rmeove it from original melData dictioanry
for key in valKeys: 
    for time in timestamps: 
        strKey = str(key)+"_"+str(time)
        if strKey in list(melData.keys()): 
            melDataVal[strKey] = melData.pop(strKey)

# Get every timestap of every sampled test song and rmeove it from original melData dictioanry
for key in testKeys: 
    for time in timestamps: 
        strKey = str(key)+"_"+str(time)
        if strKey in list(melData.keys()): 
            melDataTest[strKey] = melData.pop(strKey)

#print(len(list(melData.keys())))        # 34568 / 58 = 596
#print(len(list(melDataVal.keys())))     # 4292  / 58 = 74
#print(len(list(melDataTest.keys())))    # 4292  / 58 = 74

# Load csv with continous arousal and valence data
arousal_df = pd.read_csv("data/annotations/arousal_cont_average.csv")
valence_df = pd.read_csv("data/annotations/valence_cont_average.csv")

# Create each fo the 3 datasets
dataset = MelSpectrogramDataset(melData, arousal_df, valence_df)
val_dataset = MelSpectrogramDataset(melDataVal, arousal_df, valence_df)
test_dataset = MelSpectrogramDataset(melDataTest, arousal_df, valence_df)

# Delete extra objects that will not be used again
del melData, melDataVal, melDataTest, arousal_df, valence_df

# Create 3 samplers for each set
sampler = SongSampler(dataset)
val_sampler = SongSampler(val_dataset)
test_sampler = SongSampler(test_dataset)

# Create 3 Batch samplers to sampe each in  abatch size of 58
# This intentionally makes each batch exactly one song and every timstamp of that song
batch_sampler = BatchSampler(sampler, batch_size=len(sampler.timestamps), drop_last=False)
val_batch_sampler = BatchSampler(val_sampler, batch_size=len(val_sampler.timestamps), drop_last=False)
test_batch_sampler = BatchSampler(test_sampler, batch_size=len(test_sampler.timestamps), drop_last=False)

# Create the 3 dataloaders
data_loader = DataLoader(dataset, batch_sampler=batch_sampler)
val_data_loader = DataLoader(val_dataset, batch_sampler=val_batch_sampler)
test_data_loader = DataLoader(test_dataset, batch_sampler=test_batch_sampler)


# Variables for model parameters
# Input size is 128 for 128 mel spectorgram bins
input_size = 128
hidden_size = 20
num_layers = 2
output_size = 2

# Declare model
model = MusicEmotionLSTM(input_size, hidden_size, num_layers, output_size).to(device)

# Declare criterion for loss
criterion = nn.MSELoss()
# Uses Adam optim with leanring rate of 0.00005
optimizer = Adam(model.parameters(), lr=0.00005)

# Setup variables for early stopping
best_val_loss = np.inf
patience = 10  # Number of epochs to wait for improvement
min_delta = 0.00005 # Minimum change in loss
counter = 0 


epochs = 250

# Lists to keep track of loss statistics
losses = []
val_losses = []

i=1

# Training loop
for epoch in range(epochs): 
    # Set model to training
    model.train()
    
    # For training
    x=0
    total_loss = 0.0
    for batch in data_loader: 
        mel_data = batch['mel_data']
        arousal = batch['arousal'].float().to(device)
        valence = batch['valence'].float().to(device)
        
        optimizer.zero_grad()

        output = model(batch)
        
        val_loss = criterion(output, torch.stack((arousal, valence), dim=1))
        
        val_loss.backward()
        optimizer.step()

        total_loss += val_loss.item()

        x+=1
    
    # For validation
    val_total_loss = 0.0
    model.eval()
    y=0
    for batch in val_data_loader: 
        mel_data = batch['mel_data']
        arousal = batch['arousal'].float().to(device)
        valence = batch['valence'].float().to(device)

        output = model(batch)

        loss = criterion(output, torch.stack((arousal, valence), dim=1))

        val_total_loss += loss.item()
        y+=1

    # Compute and store avergae losses
    average_loss = total_loss / x
    losses.append(average_loss)
    
    val_average_loss = val_total_loss / y
    val_losses.append(val_average_loss)
    
    print(f"Epoch {epoch+1}\n\tAverage Train Loss: {average_loss}\n\tAverage Validation Loss: {val_average_loss}")

    # Check early stopping
    if best_val_loss - val_average_loss > min_delta:
        best_val_loss = val_average_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f'Validation loss did not improve by at least {min_delta} for {patience} epochs. Early stopping...')
        break  # Stop training

    i+=1

torch.save(model, "m_stopped.pth")

# Test dataset on model
test_loss = 0
model.eval()
z = 0
for batch in test_data_loader: 
    arousal = batch['arousal'].float().to(device)
    valence = batch['valence'].float().to(device)
    output = model(batch)
    loss = criterion(output, torch.stack((arousal, valence), dim=1))
    test_loss+=loss
    z+=1

print(f"AVG TEST LOSS (MSE): {test_loss/z}")

# Plot Loss
fig, ax = plt.subplots(figsize=(12, 4.8))

# Plotting training and validation losses on the same set of axes
ax.plot(range(1, i+1), losses, marker='o', label='Training Loss')
ax.plot(range(1, i+1), val_losses, marker='o', label='Validation Loss')

# Set titles and labels
ax.set_title('Training and Validation Loss Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Average Loss (MSE)')

# Display legend
ax.legend()

# Save and show the figure
plt.tight_layout()
plt.savefig('loss.png')
plt.show()

# Save Losses
with open('results/half_mel.json', 'w') as f:
    json.dump({"train":losses, "val":val_losses}, f)


