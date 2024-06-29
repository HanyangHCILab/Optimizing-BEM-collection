import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes=7):
        super(CNNLSTMModel, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.dropout_cnn = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the flattened dimension after CNN layers
        self.flatten_dim = 64 * 15 * 8  # Assuming input size (30, 16), which becomes (15, 8) after pooling

        # LSTM layer
        self.lstm1 = nn.LSTM(input_size=self.flatten_dim, hidden_size=256, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        self.dropout_lstm = nn.Dropout(0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Assuming input x is of shape (batch_size, seq_len, channels, height, width)
        batch_size, seq_len, C, H, W = x.size()
        
        # Process each frame in the sequence through the CNN
        cnn_out = []
        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # Extract the t-th frame
            out = F.relu(self.conv1(frame))
            out = F.relu(self.conv2(out))
            out = F.relu(self.conv3(out))
            out = self.dropout_cnn(out)
            out = self.pool(out)
            out = out.view(batch_size, -1)  # Flatten the output
            cnn_out.append(out)
        
        # Stack the CNN outputs to form the input to the LSTM
        cnn_out = torch.stack(cnn_out, dim=1)  # Shape: (batch_size, seq_len, flattened_dim)
        # Pass through LSTM
        lstm_out, _ = self.lstm1(cnn_out)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.dropout_lstm(lstm_out[:, -1, :])  # Take only the output from the last time step
        
        # Fully connected layers
        out = F.relu(self.fc1(lstm_out))
        out = self.fc2(out)
        
        return F.log_softmax(out, dim=1)
model = CNNLSTMModel()

device = "cpu"
model = model.to(device)

from torchsummaryX import summary
if __name__ == "__main__":
    summary(model, torch.rand((1, 5, 3, 30 , 16)).to(device))