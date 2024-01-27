import json
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')

file_path = 'results/loss.json'

with open(file_path, 'r') as file:
    data = json.load(file)

# Extracting data
train_loss = data['train']
val_loss = data['val']

# Plotting
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("latex/images/loss/main_loss.png", dpi=300)
plt.show()
