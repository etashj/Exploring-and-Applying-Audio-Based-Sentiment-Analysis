# Use two linear regressions to predict the next arousal and valece values isntead of an LSTM

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})
plt.style.use('seaborn-v0_8')

song_id = int(input("Enter song id: "))

arousal_df = pd.read_csv("data/annotations/arousal_cont_average.csv")
valence_df = pd.read_csv("data/annotations/valence_cont_average.csv")

a = arousal_df[arousal_df["song_id"]==song_id].iloc[0].tolist()[1:]
v = valence_df[valence_df["song_id"]==song_id].iloc[0].tolist()[1:]
time = [15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 
     20000, 20500, 21000, 21500, 22000, 22500, 23000, 23500, 24000, 24500, 
     25000, 25500, 26000, 26500, 27000, 27500, 28000, 28500, 29000, 29500, 
     30000, 30500, 31000, 31500, 32000, 32500, 33000, 33500, 34000, 34500, 
     35000, 35500, 36000, 36500, 37000, 37500, 38000, 38500, 39000, 39500, 
     40000, 40500, 41000, 41500, 42000, 42500, 43000, 43500, 44000, 44500, 45000]

data = pd.DataFrame({'time': time, 'arousal': a, 'valence': v})

# Prepare the data for linear regression
X = data[['time']]
y_arousal = data['arousal']
y_valence = data['valence']

# Fit linear regression models
model_arousal = LinearRegression()
model_arousal.fit(X, y_arousal)

model_valence = LinearRegression()
model_valence.fit(X, y_valence)

# Make predictions for the entire time range
predictions_arousal = model_arousal.predict(np.array(time).reshape(-1, 1))
predictions_valence = model_valence.predict(np.array(time).reshape(-1, 1))

# Plot the results
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.scatter(time, y_arousal, label='Actual Arousal')
plt.plot(time, predictions_arousal, label='Linear Regression Arousal', color='red')
plt.title('Arousal Linear Regression')
plt.xlabel('Time')
plt.ylabel('Arousal')
plt.legend()

plt.subplot(2, 1, 2)
plt.scatter(time, y_valence, label='Actual Valence')
plt.plot(time, predictions_valence, label='Linear Regression Valence', color='green')
plt.title('Valence Linear Regression')
plt.xlabel('Time')
plt.ylabel('Valence')
plt.legend()

plt.tight_layout()
plt.savefig("latex/images/linreg.png", dpi=300)
plt.show()