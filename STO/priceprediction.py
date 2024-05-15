# 
from STO.stock_df import df
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class LSTM(nn.Module, df):
    def __init__(self, ticker, input_size=1, hidden_layer_size=50, output_size=1):
        nn.Module.__init__()
        df.__init__(ticker)
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self):
        lstm_out, self.hidden_cell = self.lstm((self.stock_df).view(len(self.stock_df) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(self.stock_df), -1))
        self.predictions = predictions[-1]

    def processdata(self):
        # Load and preprocess the data
        stock_data = self.stock_df; stock_data = stock_data['Close'].values; stock_data = stock_data.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1)); stock_data = scaler.fit_transform(stock_data)

        # Split the data into training and testing sets
        train_split = 0.8
        split_idx = int(len(stock_data) * train_split)
        train = stock_data[:split_idx]
        train_window = 50
        test = stock_data[split_idx:]

        model = LSTM()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        epochs = 150

        for i in range(epochs):
            for seq, labels in train:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

        # Test the model
        fut_pred = 12

        test_inputs = train[-train_window:].tolist()
        model.eval()

        for i in range(fut_pred):
            seq = torch.FloatTensor(test_inputs[-train_window:])
            with torch.no_grad():
                model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))
                test_inputs.append(model(seq).item())

        actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))
        self.test = test
        self.predictions = actual_predictions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LSTM(train_window, model.hidden_layer_size, 1, 1)
    
    def calculate_loss(self):
        criterion = nn.MSELoss()
        (self.model).eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in (self.dataloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss / len(self.dataloader)