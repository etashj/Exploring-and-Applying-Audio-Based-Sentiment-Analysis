# Train simple model to predict the next value from 10 arousal and valence values (vectors)

# Import neccesary packages
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

from models import LSTMPredictionModel

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set seeds for reproducibility
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

# Define a dataset class that takes dataframes for arousal and valence and generates the dataset for every possible sequence of 10 timestamps
class TimeSeriesDataset(Dataset):
    def __init__(self, a, v): 
        # data: List of tuples of lists and ouptut tuple
        #       [ ([ ( ), ( ), ... ], (output tuple) )), ([ input seq of tuples ], ), ...]
        self.a = a
        self.v = v
        
        # INPUT LENGTH
        self.INPUT_LEN = 10
        self.data = []
        print("Loading Data")
        for index, row in a.iterrows(): 
            for i in range(1, len(a.columns)-1-self.INPUT_LEN): 
                inList = row[i:i+10]
                inList = torch.from_numpy(inList.to_numpy()).float().to(device)
                inList = torch.stack((inList, torch.from_numpy(v.iloc[index][i:i+10].to_numpy()).float().to(device)), dim=0)
                out = torch.stack((torch.tensor([row[i+11]]).float().to(device), torch.tensor([v.iloc[index][i+11]]).float().to(device)), dim=0)
                inList = torch.transpose(inList, 0, 1)
                self.data.append((inList, out))
        print("Data Loading Complete")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        ## tuple of input seq and output tensor

# Open csv into dataframes
arousal_df = pd.read_csv("data/annotations/arousal_cont_average.csv")
valence_df = pd.read_csv("data/annotations/valence_cont_average.csv")

# Create dataset
dataset = TimeSeriesDataset(arousal_df, valence_df)

# Split the dataset into train, validation, and test
trainSet, valSet, testSet = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

# Create 3 dataloaders with batch of 64 and shuffle true
dataloader = DataLoader(trainSet, batch_size=64, shuffle=True)
valDataloader = DataLoader(valSet, batch_size=64, shuffle=True)
testDataloader = DataLoader(testSet, batch_size=64, shuffle=True)

# Define model parameters 
# Input is 2 for arousal and valence with 10 timesteps beacuse of INPUT_LEN
input_size = 2
hidden_size = 32
output_size = 2

# Define model, criterion (MSE), and Adam optimizer (lr=0.0001)
model = LSTMPredictionModel(input_size, hidden_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.0001)

num_epochs = 10

# Lists to keep track of loss statistics
losses=[]
val_losses=[]

# Training loop
for epoch in range(num_epochs):
     # Set model to training
    model.train()

    # For training
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets.squeeze())
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    losses.append(loss.item())
    
    # For validation
    model.eval()
    val_tot = 0
    y=0
    for inputs, targets in valDataloader: 
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets.squeeze())
        val_tot+=loss.item()
        y+=1

    val_losses.append(val_tot/y)




torch.save(model, "models/predictor.pth")

test_loss = 0
model.eval()
z = 0
for inputs, targets in testDataloader: 
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets.squeeze())
    test_loss+=loss
    z+=1

print(f"AVERAGE TRAIN LOSS: {sum(losses)/len(losses)}")
print(f"AVERAGE VAL LOSS: {sum(val_losses)/len(val_losses)}")
print(f"AVG TEST LOSS (MSE): {test_loss/z}")

# Plot Loss
fig, ax = plt.subplots(figsize=(12, 4.8))

# Plotting training and validation losses on the same set of axes
ax.plot(range(1, num_epochs+1), losses, marker='o', label='Training Loss')
ax.plot(range(1, num_epochs+1), val_losses, marker='o', label='Validation Loss')

# Set titles and labels
ax.set_title('Training and Validation Loss Over Epochs')
ax.set_xlabel('Epoch')
ax.set_ylabel('Average Loss')

# Display legend
ax.legend()

# Save and show the figure
plt.tight_layout()
plt.savefig('loss.png')
plt.show()

# Save Losses
with open('results/predictor.json', 'w') as f:
    json.dump({"train":losses, "val":val_losses}, f)