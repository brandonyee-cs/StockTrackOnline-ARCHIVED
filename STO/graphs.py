from STO.priceprediction import LSTM
import matplotlib.pyplot as plt

class graphShow(LSTM):
    def __init__(self):
        super().__init__()

    def plot_close(self):
        plt.figure(figsize=(14,7))
        plt.plot(self.stock_df['Close'])
        plt.title('Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()

    def plot_predictions(self):
        LSTM.forward(); LSTM.processdata()
        plt.plot(self.test, color='red',label='Real Stock Price')
        plt.plot(self.predictions, color='blue',label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show() 