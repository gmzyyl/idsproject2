# model_defs.py
import torch
import torch.nn as nn

class ANN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(ANN_Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        self.act = activation_fn
    def forward(self, x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        return self.fc3(x)

class CNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        linear_input = 64 * (input_size // 2) 
        self.fc1 = nn.Linear(linear_input, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.act = activation_fn
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_fn):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.act = activation_fn
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.act(self.fc1(out))
        return self.fc2(out)