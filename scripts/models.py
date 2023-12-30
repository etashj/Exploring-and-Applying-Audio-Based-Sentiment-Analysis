## HOSTS BOTH MODELS SO THEY CAN BE IMPORTED

import torch
import torch.nn as nn

class MusicEmotionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropoutP = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropoutP)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropoutP)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, (hidden_state, cell_state) = self.lstm(x['mel_data'])
        lstm_out2, (hidden_state2, cell_state2) = self.lstm2(lstm_out)
        last_output = lstm_out2[:, -1, :]
        output = self.linear(last_output)
        return output
    
class LSTMPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMPredictionModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Taking the last time step's output
        return output
