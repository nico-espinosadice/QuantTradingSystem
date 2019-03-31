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
    self.best_svm_rbf, self.best_svm_poly = {}, {}
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

    df = pd.read_csv(symbol + '.csv', infer_datetime_format=True)
    df = df.rename(columns={'Unnamed: 0': 'Date'})

    df.to_csv(symbol + '.csv', encoding='utf-8')

    self.data[symbol] = pd.read_csv(csv_path, infer_datetime_format=True)

  def make_train_cv_test(self, symbol, split, N, csv_path):
    self.update_data(symbol=symbol, csv_path=csv_path) # get most current data

    this_data = self.data[symbol].drop(['Date', 'Unnamed: 0'], axis=1) # get rid of date column for the data to be trained

    this_data = this_data.dropna() # gets rid of last row with only open data
    # X = np.array(this_data)
    X = np.array(this_data.ewm(span=N, adjust=False).mean())  # all of the exponential moving averages
    startrow = np.zeros((1, X[0].size)) # make the first row unknown
    X = np.append(startrow, X, axis=0)

    # Scaling data
    scaler = MinMaxScaler(copy=False)
    scaler.fit(X)
    scaler.transform(X)

    opens = np.array(this_data['Open'])
    y = np.array([1 if (opens[i] < opens[i+1]) else 0 for i in range(opens.size-1)])
    y = np.append(y, 0)

    train = int(self.data[symbol]['Open'].size * split[0])  # gives number of training days
    cv = int(self.data[symbol]['Open'].size * split[1])  # number of cross validation days
    test = int(self.data[symbol]['Open'].size * split[2]) -1  # gives number of testing days (doesn't include last point)

    for i in range(len(X)-1):
      X[i][0] = X[i+1][0] # using current opening data
    X[-1][0] = 0

    # Makes train, cv, and test data for the symbol. Also does not include
    # the last day of data in test_data, because there is not a label for it.
    self.train_data[symbol], self.train_labels[symbol] = X[:train], y[:train]
    self.cv_data[symbol], self.cv_labels[symbol] = X[train:train + cv], y[train:train + cv]
    self.test_data[symbol], self.test_labels[symbol] = X[train + cv: train + cv + test], y[train + cv:train + cv + test]
    print(self.train_data[symbol][:10])
    print(self.train_labels[symbol][:10])
    mornings = np.array(self.data[symbol]['Open']) # gets all opens
    self.train_prices[symbol], self.cv_prices[symbol], self.test_prices[symbol] = \
      mornings[:train], mornings[train:train+cv], mornings[train+cv:-1]

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

    pc, amounts_rbf, Cs = [], [], [] # lists for percent correct (pc) and C values (Cs)
    amount_to_svm_rbf = {} # maps from the amount a svm would make to the svm itself
    start, end = c_test_range
    c = (end/start) ** (1/num_iterations)
    Cs = [start*(c**i) for i in range(1, num_iterations+1)]
    for i in range(1, num_iterations+1): # cycle through different possible C values
      svm_rbf = svm.SVC(C=Cs[i-1], kernel='rbf', gamma='auto')
      svm_rbf.fit(self.train_data[symbol], self.train_labels[symbol])
      amount_made = self.simulate_amount_gained(symbol, svm_rbf, type='cv')
      amounts_rbf += [amount_made]
      if amount_made not in amount_to_svm_rbf.keys():
        amount_to_svm_rbf[amount_made] = svm_rbf
      pc += [self.percent_correct(svm_rbf, 'cv', symbol=symbol)]

    most_made_rbf = max(amount_to_svm_rbf.keys())
    self.cv_performance.add( (symbol, most_made_rbf) )
    self.best_svm_rbf[symbol] = amount_to_svm_rbf[most_made_rbf]

    pc_poly, amounts_poly, Cs = [], [], []  # lists for percent correct (pc) and C values (Cs)
    amount_to_svm_poly = {}  # maps from the amount a svm would make to the svm itself
    start, end = c_test_range
    c = (end / start) ** (1 / num_iterations)
    Cs = [start * (c ** i) for i in range(1, num_iterations + 1)]
    for i in range(1, num_iterations + 1):  # cycle through different possible C values
      svm_poly = svm.SVC(C=Cs[i - 1], kernel='poly', gamma='auto')
      svm_poly.fit(self.train_data[symbol], self.train_labels[symbol])
      amount_made = self.simulate_amount_gained(symbol, svm_poly, type='cv')
      amounts_poly += [amount_made]
      if amount_made not in amount_to_svm_poly.keys():
        amount_to_svm_poly[amount_made] = svm_poly
      pc_poly += [self.percent_correct(svm_poly, 'cv', symbol=symbol)]

    most_made_poly = max(amount_to_svm_poly.keys())
    self.cv_performance.add((symbol, most_made_poly))
    self.best_svm_poly[symbol] = amount_to_svm_poly[most_made_poly]

    plt.plot(np.log(Cs), amounts_rbf, label='RBF')
    plt.plot(np.log(Cs), amounts_poly, label='polynomial')
    plt.title(symbol)
    plt.xlabel('C values')
    plt.ylabel('Amount made with this C value')
    plt.legend()
    plt.show()



  def simulate_amount_gained_buy_always(self, symbol, type='test', show_output=False):
    '''
    :param symbol: stock symbol, as a string
    :param type: either test or cv
    :return: the amount that would be gained from just buying every day
    '''
    print('\nBUY ONLY STRATEGY')
    made = self.simulate_amount_gained(symbol=symbol, svm1=None, svm2=None, type=type+' buy', show_output=show_output)
    print('END BUY ONLY STRATEGY\n')
    return made

  def simulate_amount_gained(self, symbol, svm1, svm2=None, type='test', show_output=False):
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
      predictions = svm1.predict(self.test_data[symbol])
      if svm2 != None:
        predictions2 = svm2.predict(self.test_data[symbol])
        predictions = [1 if (predictions[i] == 1 or predictions2[i] == 1) else 0 for i in range(predictions.size)]
      prices = self.test_prices[symbol].flatten()  # flattens 2D array of prices
    elif type == 'cv':
      predictions = svm1.predict(self.cv_data[symbol])
      if svm2 != None:
        predictions2 = svm2.predict(self.cv_data[symbol])
        predictions = [1 if (predictions[i] == 1 or predictions2[i] == 1) else 0 for i in range(predictions.size)]
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
      my_performance = 100*self.percent_correct(self.best_svm_rbf[symbol], 'test', symbol=symbol)
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
                                svm1=self.best_svm_rbf[symbol],
                                type='test', show_output=True)
    buy_always_amount_gained = self.simulate_amount_gained_buy_always(symbol, type='test', show_output=True)
    print('Buy always amount: ${}'.format(buy_always_amount_gained))
    print('Advantage: ${}'.format(amount_gained-buy_always_amount_gained))

if __name__ == '__main__':
  s = SVM_trader(2000, True)
  # stock_symbols = ['ZNGA', 'SPY', 'AAPL', 'IBM']
  stock_symbols = ['SPY']
  for stock_symbol in stock_symbols:
    print('\n\n' + stock_symbol)
    s.run_test(symbol=stock_symbol, N=20, split=(.4, .1, .4), path=stock_symbol+'.csv',
               c_test_range=[1, 6000000], num_iterations=10)

