# Import necessary libraries
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import alpaca_trade_api as tradeapi

class RNN:
    def __init__(self, cash_available, csv_train, csv_test, stock):
        self.cash_available = cash_available
        self.csv_train = csv_train
        self.csv_test = csv_test
        self.stock = stock

        self.model_json = "rnn.json"

        self.date = "2017-00-00T00:00:00.000Z" # date specified here only for testing purposes

        self.sc = MinMaxScaler(feature_range=(0, 1))
        self.api = tradeapi.REST(key_id='PKXVRTVRHQWTL50AYFKA',
                                 secret_key='',
                                 base_url='https://paper-api.alpaca.markets')

        self.dataset, self.training_set, self.X_train, self.y_train, \
            self.regressor, self.dataset_test, self.test_set, self.real_stock_price, \
            self.inputs, self.dataset_total, self.X_test, self.predicted_stock_price, \
            self.data = [], [], [], [], [], [], [], [], [], [], [], [], []

    def loadTrainData(self):
        """ Loads data for training. """
        self.dataset = pd.read_csv(self.csv_train, index_col="Date", parse_dates=True)

        training_set = self.dataset["Open"]
        self.training_set = pd.DataFrame(training_set)

    def loadTestData(self):
        """ Loads the data for testing. """
        self.dataset_test = pd.read_csv(self.csv_test, index_col="Date", parse_dates=True)
        self.real_stock_price = self.dataset_test.iloc[:, 1:2].values
        self.test_set = self.dataset_test['Open']
        self.test_set = pd.DataFrame(self.test_set)

    def updateData(self):
        self.stock_dataset = self.api.get_barset(self.stock, "1D", start=self.date)
        self.stock_data = self.stock_dataset[self.stock]

        for data_row in self.stock_data:
            date = data_row.t
            open = data_row.o
            high = data_row.h
            low = data_row.l
            close = data_row.c
            volume = data_row.v

            df_row = pd.DataFrame(np.array([[date, open, high, low, close, volume]]), columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df_row = df_row.set_index('Date')

            self.dataset = self.dataset.append(df_row)

    def scaleTrainData(self):
        """ Scales the training data using sklearn MinMaxScaler. """
        self.loadTrainData()

        # Feature Scaling
        self.training_set_scaled = self.sc.fit_transform(self.training_set)

        # Creating a data structure with 60 timesteps and 1 output
        self.X_train = []
        self.y_train = []
        for i in range(60, 1258):
            self.X_train.append(self.training_set_scaled[i-60:i, 0])
            self.y_train.append(self.training_set_scaled[i, 0])
        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        # Reshaping
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 1))

    def buildRNN(self):
        """ Builds the RNN. """
        self.scaleTrainData()

        # Initialising the RNN
        self.regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50, return_sequences=True,
                        input_shape=(self.X_train.shape[1], 1)))
        self.regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50))
        self.regressor.add(Dropout(0.2))

        # Adding the output layer
        self.regressor.add(Dense(units=1))

    def fitRNN(self):
        """ Fits the RNN to the data. """
        self.buildRNN()

        # Compiling the RNN
        self.regressor.compile(optimizer='adam', loss='mean_squared_error')

        # Fitting the RNN to the Training set
        self.regressor.fit(self.X_train, self.y_train, epochs=100, batch_size=32)

    def getPredictions_TestDataset(self):
        """ Gets predictions for the test dataset. """
        self.loadTestData()
        self.fitRNN()

        # Getting the predicted stock price
        self.dataset_total = pd.concat((self.dataset['Open'], self.dataset_test['Open']), axis=0)

        self.inputs = self.dataset_total[len(self.dataset_total) - len(self.dataset_test) - 60:].values
        self.inputs = self.inputs.reshape(-1, 1)
        self.inputs = self.sc.transform(self.inputs)

        for i in range(60, len(self.dataset_test) + 60):
            self.X_test.append(self.inputs[i-60:i, 0])

        self.X_test = np.array(self.X_test)
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0], self.X_test.shape[1], 1))

        #self.predicted_stock_price = self.regressor.predict(self.X_test)
        #self.predicted_stock_price = self.sc.inverse_transform(self.predicted_stock_price)

    def visualize(self):
        """ Visualize the results of the real stock price versus the predicted stock price. """
        plt.plot(self.real_stock_price, color='red', label='Real Google Stock Price')
        plt.plot(self.predicted_stock_price, color='blue',
                label='Predicted Google Stock Price')
        plt.title('Google Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Google Stock Price')
        plt.legend()
        plt.xlim(0, 30)
        plt.ylim(750, 875)
        plt.show()

    def predictTomorrow(self):
        """ Predicts the stock price for tomorrow. """
        self.loadTrainData()
        self.updateData()
        self.fitRNN()

        self.inputs = self.dataset[:].values
        self.inputs = self.inputs.reshape(-1, 1)
        self.inputs = self.sc.transform(self.inputs)

        self.inputs = np.array(self.inputs)
        self.inputs = np.reshape(self.inputs, (self.inputs.shape[0], self.inputs.shape[1], 1))

        self.predicted_stock_price = self.regressor.predict(self.inputs)
        self.predicted_stock_price = self.sc.inverse_transform(self.predicted_stock_price)

    def saveModel(self):
        """ Saves RNN to JSON file """
        self.model_json = self.regressor.to_json() # serialize model to JSON
        with open("model.json", "w") as json_file:
            json_file.write(self.model_json)
        self.regressor.save_weights("model.h5")  # serialize weights to HDF5

    def loadModel(self):
        self.json_file = open('model.json', 'r') # load json and create model
        self.loaded_model_json = self.json_file.read()
        self.json_file.close()
        self.regressor = model_from_json(self.loaded_model_json)
        self.regressor.load_weights("model.h5") # load weights into new model

    def buy(self):
        # Submit a market order to buy 1 share of stock at market price
        self.api.submit_order(
            symbol=self.stock,
            qty=1,
            side='buy',
            type='market',
            time_in_force='gtc'
        )

    def sell(self):
        # Submit a market order to sell 1 share of stock at market price
        self.api.submit_order(
            symbol=self.stock,
            qty=1,
            side='sell',
            type='market',
            time_in_force='gtc'
        )
