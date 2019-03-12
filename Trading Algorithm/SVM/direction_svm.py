import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import csv


class SVM_trader:
  def __init__(self, startamount, short):
    self.starting_amount = startamount
    self.can_short = short
    self.best_svm = {}
    self.data = {}
    self.train_data, self.cv_data, self.test_data,\
      self.train_labels, self.cv_labels, self.test_labels,\
      self.train_prices, self.cv_prices, self.test_prices= \
      {}, {}, {}, {}, {}, {}, {}, {}, {}
    self.cv_performance = set() # a set of tuples, where tup[0] = stock_name, tup[1] = most the stock made in cross validation

  def update_data(self, symbol, csv_path):
    '''
    Purpose:
      Updates with historical data up to this day
    :param symbol: name of the symbol to update
    :param csv_path: string path to the csv file for that symbol
    :return: True if it found the file and successfully appended,
             otherwise false
    '''

    API_URL = "https://www.alphavantage.co/query"
    data = {
      "function": "TIME_SERIES_DAILY",
      "symbol": symbol,
      "outputsize": "full",
      "datatype": "json",
      "apikey": "A1A3K3EDC9CG3LVW",
    }
    response = requests.get(API_URL, params=data)

    data = response.json()['Time Series (Daily)']
    df = pd.DataFrame(data).transpose()

    df = df[::-1]  # goes from earliest to newest

    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.to_csv(symbol + '.csv', encoding='utf-8')
    nan = float('NaN')

    df = pd.read_csv(symbol + '.csv', infer_datetime_format=True)
    df = df.rename(columns={'Unnamed: 0': 'Date'})

    df.at[df.shape[0] - 1, 'High'] = nan
    df.at[df.shape[0] - 1, 'Low'] = nan
    df.at[df.shape[0] - 1, 'Close'] = nan
    df['Volume'] = np.asarray(np.array(df['Volume']), 'float')
    df.at[df.shape[0] - 1, 'Volume'] = nan

    df.to_csv(symbol + '.csv', encoding='utf-8')

    self.data[symbol] = pd.read_csv(csv_path, infer_datetime_format=True)

  def make_train_cv_test(self, symbol, split, N, csv_path):
    self.update_data(symbol=symbol, csv_path=csv_path) # get most current data

    this_data = self.data[symbol].drop(['Date'], axis=1) # get rid of date column for the data to be trained
    this_data = this_data.dropna() # gets rid of last row with only open data
    X = np.array(this_data.ewm(span=N, adjust=False).mean())  # all of the exponential moving averages
    startrow = np.zeros((1, X[0].size)) # make the first row unknown
    X = np.append(startrow, X, axis=0)
    this_morning = np.array(self.data[symbol]['Open']) # gets all opens
    this_morning = this_morning.reshape(-1, 1)
    X = np.column_stack((this_morning, X))
    X = X[:-1] # gets rid of the last data value; no need to use it

    # Scaling data
    scaler = MinMaxScaler(copy=False)
    scaler.fit(X)
    scaler.transform(X)

    opens, closes = np.array(this_data['Open']), np.array(this_data['Close'])
    assert opens.size == closes.size
    y = np.array([1 if (opens[i] < closes[i]) else 0 for i in range(opens.size)])

    train = int(self.data[symbol]['Open'].size * split[0])  # gives number of training days
    cv = int(self.data[symbol]['Open'].size * split[1])  # number of cross validation days

    self.train_data[symbol], self.train_labels[symbol] = X[:train], y[:train]
    self.cv_data[symbol], self.cv_labels[symbol] = X[train:train + cv], y[train:train + cv]
    self.test_data[symbol], self.test_labels[symbol] = X[train + cv:], y[train + cv:]

    self.train_prices[symbol], self.cv_prices[symbol], self.test_prices[symbol] = \
      this_morning[:train], this_morning[train:train+cv], this_morning[train+cv:]

  def percent_correct(self, s_v_m, type, symbol):
    if(type == 'cv'):
      predictions = s_v_m.predict(self.cv_data[symbol])
      labels = self.cv_labels[symbol]
    elif(type == 'test'):
      predictions = s_v_m.predict(self.test_data[symbol])
      labels = self.test_labels[symbol]
    else:
      assert False, 'Enter a valid type'
    num_correct = 0
    for i, label in enumerate(labels): num_correct += 1 if (label == predictions[i]) else 0
    return float(num_correct) / len(labels)  # givest the accuracy as a percentage [0, 1]

  def make_svm(self, symbol, N, path, split=(.6, .2, .2), c_test_range=[1, 65000], num_iterations=20):
    self.make_train_cv_test(symbol, split, N, path) # prepare the data

    pc, amounts, Cs = [], [], [] # lists for percent correct (pc) and C values (Cs)
    amount_to_svm = {} # maps from the amount a svm would make to the svm itself
    start, end = c_test_range
    c = (end/start) ** (1/num_iterations)
    Cs = [start*(c**i) for i in range(1, num_iterations+1)]
    for i in range(1, num_iterations+1): # cycle through different possible C values
      support_vector_machine = svm.SVC(C=Cs[i-1], kernel='poly', gamma='auto')
      support_vector_machine.fit(self.train_data[symbol], self.train_labels[symbol])
      amount_made = self.simulate_amount_gained(symbol, support_vector_machine, type='cv')
      amounts += [amount_made]
      if amount_made in amount_to_svm.keys():
        # print('Cross validation yields same answer for another C value')
        pass
      else:
        amount_to_svm[amount_made] = support_vector_machine
      pc += [self.percent_correct(support_vector_machine, 'cv', symbol=symbol)]
    plt.plot(np.log(Cs), amounts)
    plt.title(symbol)
    plt.xlabel('C values')
    plt.ylabel('Amount made with this C value')
    plt.show()

    most_made = max(amount_to_svm.keys())
    self.cv_performance.add( (symbol, most_made) )
    self.best_svm[symbol] = amount_to_svm[most_made]

  def simulate_amount_gained_buy_always(self, symbol, type='test', show_output=False):
    '''
    :param symbol: stock symbol, as a string
    :param type: either test or cv
    :return: the amount that would be gained from just buying every day
    '''
    print('\nBUY ONLY STRATEGY')
    made = self.simulate_amount_gained(symbol, None, type+' buy', show_output=show_output)
    print('END BUY ONLY STRATEGY\n')
    return made

  def simulate_amount_gained(self, symbol, svm, type='test', show_output=False):
    '''
    :param: symbol: the name of the stock (string)
    :param: svm: the support vector machine (sk learn SVM)
    :param: type: either 'cv', 'test', 'cv buy', 'test buy'
    :return: the amount of money that would be made if 1 share
    of the company's stock was bought every time the algorithm
    predicted 'buy' and 1 share of the company's stock was
    sold every time the algorithm predicted 'sell'. This
    assumes no spending cap, and that the stock can be shorted
    indefinitely. Assumes buying and selling at open.
    '''
    cash, num_stocks = self.starting_amount, 0  # starts at 0 stocks
    if type == 'test':
      predictions = svm.predict(self.test_data[symbol])
      prices = self.test_prices[symbol].flatten() # flattens 2D array of prices
    elif type == 'cv':
      predictions = svm.predict(self.cv_data[symbol])
      prices = self.cv_prices[symbol].flatten()  # flattens 2D array of prices
    elif type == 'test buy':
      predictions = np.array([1 for datapoint in self.test_data[symbol]])
      prices = self.test_prices[symbol].flatten() # flattens 2D array of prices
    elif type == 'cv buy':
      predictions = np.array([1 for datapoint in self.cv_data[symbol]])
      prices = self.test_prices[symbol].flatten()  # flattens 2D array of prices

    for i, prediction in enumerate(predictions):
      if prediction == 1 and prices[i] < cash:
        num_stocks_to_buy = int(cash/prices[i])
        num_stocks += num_stocks_to_buy
        cash = cash % prices[i]
      elif prediction == 0 and (self.can_short or num_stocks >= 1):
        cash += prices[i] * num_stocks
        num_stocks = 0

    num_days = len(predictions)
    stock_capital = num_stocks * prices[-1]
    money_made = stock_capital + cash - self.starting_amount
    if show_output == True :
      print('Ends with ${} in capital'.format(cash))
      print('Ends with ${} in stocks'.format(stock_capital))
      print('Ends with net of ${}'.format(money_made))
      print('Made, on average, ${} per day over {} days'.format(
        money_made / num_days,
        num_days
      ))
    return money_made

  def run_test(self, symbol, N, split, path, c_test_range=[1, 65000], num_iterations=20):
    '''
    :param symbol: stock tickername
    :param N: parameter for exponential moving average
    :param split: this is a tuple that should sum to 1, denoting the proportion of train, cv, and test data to use.
      for example, split=(.6, .2, .2) would use 60% of the data to train, 20% to do cross validation, and 20% to test.
    :param path: a string that gives the path to the .csv file with price data
    :param c_test_range: the range of values used for the C parameter to the SVM
    :param num_iterations: how many times to check different C values
    :return: no return, but prints out performance to log
    '''

    self.make_svm(symbol, N, path, split, c_test_range, num_iterations)
    with open('SVM_performance.csv', 'a') as csvfile:
      writer = csv.writer(csvfile)
      guess_buy_performance = 100*(np.sum(self.test_labels[symbol])/np.size(self.test_labels[symbol]))
      my_performance = 100*self.percent_correct(self.best_svm[symbol], 'test', symbol=symbol)
      my_adjusted_performance = my_performance-guess_buy_performance
      writer.writerow([symbol, N, -1, -1,
                      my_performance,
                      guess_buy_performance,
                      my_adjusted_performance])
    csvfile.close()
    print('My strategy percent correct: {}%'.format(my_performance))
    print('Always buy strategy percent correct: {}%'.format(guess_buy_performance))
    print('Advantage percent correct: {}%'.format(my_adjusted_performance))
    amount_gained = self.simulate_amount_gained(symbol=symbol,
                                svm=self.best_svm[symbol],
                                type='test', show_output=True)
    buy_always_amount_gained = self.simulate_amount_gained_buy_always(symbol, type='test', show_output=True)
    print('Buy always amount: ${}'.format(buy_always_amount_gained))
    print('Advantage: ${}'.format(amount_gained-buy_always_amount_gained))

if __name__ == '__main__':
  s = SVM_trader(2000, False)
  stock_symbols = ['SPY', 'AAPL', 'ZNGA', 'IBM']
  for stock_symbol in stock_symbols:
    s.run_test(symbol=stock_symbol, N=10, split=(.4, .2, .4), path=stock_symbol+'.csv',
               c_test_range=[1, 6000000], num_iterations=10)

