import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import csv


class SVM_trader:
  def __init__(self):
    self.api = tradeapi.REST (
      key_id='PKOH1X3RLHOUW294ZQ0S',
      secret_key='H7RgDXnwF18cvvDE3yxm1ZjX/B816tPlpMHcm8Je',
      base_url='https://paper-api.alpaca.markets'
    )

  def run_test(self, symbol, N, train_barriers, test_barriers, cv_barriers, c_test_range=[1, 65000], num_iterations=20):
    num_years = (test_barriers[1]-train_barriers[0]).days / 365.25
    opens, closes, highs, lows, volumes = [], [], [], [], []
    end = train_barriers[0]
    for i in range(round(num_years)):
      start = end
      end = pd.Timestamp(train_barriers[0].year+1+i,
                         train_barriers[0].month,
                         train_barriers[0].day)
      bars = self.api.get_barset(symbol, 'day',
                            start = start, end=end,
                            limit=365)[symbol]
      opens += [item.o for item in bars]
      closes += [item.c for item in bars]
      highs += [item.h for item in bars]
      lows += [item.l for item in bars]
      volumes += [item.v for item in bars]


    remaining_days = (test_barriers[1]-end).days
    bars = self.api.get_barset(symbol, 'day', start=end, end=test_barriers[1], limit=remaining_days)[symbol]
    opens += [item.o for item in bars]
    closes += [item.c for item in bars]
    highs += [item.h for item in bars]
    lows += [item.l for item in bars]
    volumes += [item.v for item in bars]

    # EMA of the data
    data= pd.DataFrame(np.column_stack((opens, closes, highs, lows, volumes)))
    X = np.array(data.ewm(span=N, adjust=False).mean()) # all of the exponential moving averages
    # Scaling data
    scaler = MinMaxScaler(copy=False)
    scaler.fit(X)
    scaler.transform(X)
    MinMaxScaler(X, copy=False) # normalizes X in-place

    # Making SVM
    ### Making the labels array (should have bought on day[i] if labels[i] < labels[i+1]
    y = [1 if (opens[i] < opens[i + 1]) else 0 for i in range(0, len(opens) - 1)]
    y += [0] # guesses 'sell' for last prediction
    y = np.array(y)

    train = (train_barriers[1]-train_barriers[0]).days # number of training days
    cv = (cv_barriers[1]-cv_barriers[0]).days # number of cross validation days

    train_data, train_labels = X[:train], y[:train]
    cv_data, cv_labels = X[train:train + cv], y[train:train + cv]
    test_data, test_labels = X[train + cv:], y[train + cv:]

    def percent_correct(s_v_m, type):
      predictions = s_v_m.predict(cv_data) if(type=='cv') else s_v_m.predict(test_data)
      num_correct = 0
      labels = cv_labels if(type=='cv') else test_labels
      for i, label in enumerate(labels): num_correct += 1 if (label == predictions[i]) else 0
      return float(num_correct)/len(labels) # givest the accuracy as a percentage [0, 1]

    def amount_gained(s_v_m, prices):
      '''
      :param s_v_m: the already trained support vector machine
      :param prices: an array with the opening prices on the days
      in question (prices[i] is price on ith day of s_v_m prediction)
      :return: the amount of money that would be made if 1 share
      of the company's stock was bought every time the algorithm
      predicted 'buy' and 1 share of the company's stock was
      sold every time the algorithm predicted 'sell'. This
      assumes no spending cap, and that the stock can be shorted
      indefinitely. Assumes buying and selling at open.
      '''
      most_money_invested = 0
      paper_capital, num_stocks = 0, 0 # starts at 0 of both
      predictions = s_v_m.predict(test_data)
      for i, prediction in enumerate(predictions):
        if(-paper_capital > most_money_invested):
          most_money_invested = -paper_capital
        if(prediction == 1):
          num_stocks += 1
          paper_capital -= prices[i]
        elif(prediction == 0):
          num_stocks -= 1
          paper_capital += prices[i]

      num_days = len(predictions)
      stock_capital = num_stocks*prices[-1]
      money_made = stock_capital+paper_capital
      print('Ends with ${} in capital'.format(paper_capital))
      print('Ends with ${} in stocks'.format(stock_capital))
      print('Ends with net of ${}'.format(money_made))
      print('${} invested at maximum'.format(most_money_invested))
      print('Made, on average, ${} per day over {} days'.format(
        money_made/num_days,
        num_days
      ))
      return stock_capital + paper_capital


    def simulate_amount_gained(s_v_m, prices, starting_amount, can_short=False):
      '''
      :param s_v_m: the already trained support vector machine
      :param prices: an array with the opening prices on the days
      in question (prices[i] is price on ith day of s_v_m prediction)
      :return: the amount of money that would be made if 1 share
      of the company's stock was bought every time the algorithm
      predicted 'buy' and 1 share of the company's stock was
      sold every time the algorithm predicted 'sell'. This
      assumes no spending cap, and that the stock can be shorted
      indefinitely. Assumes buying and selling at open.
      '''
      cash, num_stocks = starting_amount, 0 # starts at 0 stocks
      predictions = s_v_m.predict(test_data)
      for i, prediction in enumerate(predictions):
        if prediction == 1 and prices[i] < cash:
          num_stocks += 1
          cash -= prices[i]
        elif prediction == 0 and (can_short or num_stocks >= 1):
            num_stocks -= 1
            cash += prices[i]

      num_days = len(predictions)
      stock_capital = num_stocks*prices[-1]
      money_made = stock_capital+cash-starting_amount
      print('Ends with ${} in capital'.format(cash))
      print('Ends with ${} in stocks'.format(stock_capital))
      print('Ends with net of ${}'.format(money_made))
      print('Made, on average, ${} per day over {} days'.format(
        money_made/num_days,
        num_days
      ))
      return money_made

    pc, Cs = [], []
    start, end = c_test_range
    c = (end/start) ** (1/num_iterations)
    pc_to_c = {}
    for i in range(1, num_iterations+1):
      Cs += [start*(c**i)]
      support_vector_machine = svm.SVC(C=Cs[-1], kernel='rbf', gamma='auto')
      support_vector_machine.fit(train_data, train_labels)
      pc += [percent_correct(support_vector_machine, 'cv')]
      pc_to_c[pc[-1]] = Cs[-1]
    plt.plot(np.log(Cs), pc)
    plt.show()

    with open('SVM_performance.csv', 'a') as csvfile:
      writer = csv.writer(csvfile)
      best_c = pc_to_c[np.max(pc)]
      best_svm = svm.SVC(C=best_c, kernel='rbf', gamma='auto')
      best_svm.fit(train_data, train_labels)
      guess_buy_performance = 100*(np.sum(y)/np.size(y))
      my_performance = 100*percent_correct(best_svm, 'test')
      my_adjusted_performance = my_performance-guess_buy_performance
      writer.writerow([symbol, N, best_c, num_years,
                      my_performance,
                      guess_buy_performance,
                      my_adjusted_performance])
    csvfile.close()
    print('My strategy: {}%'.format(my_performance))
    print('Always buy strategy: {}%'.format(guess_buy_performance))
    print('Advantage: {}%'.format(my_adjusted_performance))
    simulate_amount_gained(best_svm, opens[1:], 2000, can_short=False)

if __name__ == '__main__':

  start_train = pd.Timestamp(2012, 2, 1)
  end_train = pd.Timestamp(2015, 2, 10)
  start_cv = pd.Timestamp(2016, 2, 10)
  end_cv = pd.Timestamp(2017, 2, 20)
  start_test = pd.Timestamp(2018, 2, 20)
  end_test = pd.Timestamp(2019, 2, 24)
  s = SVM_trader()
  s.run_test(symbol='GOOG', N=2, train_barriers=[start_train, end_train], test_barriers=[start_test, end_test], c_test_range=[1, 600000], num_iterations=10, cv_barriers=[start_cv, end_cv])

