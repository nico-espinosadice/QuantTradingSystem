import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def ema(num, values):
  '''
  :param alpha: self explanatory
  :param values: the (perhaps price) values (as a list)
  :param last: the default last value, before the list begins
  :return: a list of size N, where the ith entry is the ema for values
           up to, but not indluding, i.
  This works by
  '''
  alpha = 2/(num+1)

  last = values[0]
  emas = [last]
  for i in range(1, len(values)):
    emas += [last + alpha * (values[i] - last)]
    last = emas[-1]
  return np.array(emas)

api = tradeapi.REST(
    key_id='PKOH1X3RLHOUW294ZQ0S',
    secret_key='H7RgDXnwF18cvvDE3yxm1ZjX/B816tPlpMHcm8Je',
    base_url='https://paper-api.alpaca.markets'
)

num_years = 1
symbol = 'GOOG'
N = 5
opens, closes, highs, lows, volumes = [], [], [], [], []
end = pd.Timestamp(2019-num_years, 1, 1, 12)
for i in range(num_years):
  start = end
  end = pd.Timestamp(2019-num_years+1+i, 1, 1, 12)
  apple_bars = api.get_barset(symbol, 'day', start = start, end=end, limit=365)[symbol]
  opens += [item.o for item in apple_bars]
  closes += [item.c for item in apple_bars]
  highs += [item.h for item in apple_bars]
  lows += [item.l for item in apple_bars]
  volumes += [item.v for item in apple_bars]

  # EMA of the data
ema_opens = ema(N, opens)
ema_closes = ema(N, closes)
ema_highs = ema(N, highs)
ema_lows = ema(N, lows)
ema_volumes = ema(N, volumes)

# Making single data array
X = np.column_stack((ema_opens, ema_closes, ema_highs, ema_lows, ema_volumes))
# X = np.column_stack((ema_opens, ema_highs, ema_lows, ema_volumes))
# X = ema_opens.reshape(-1, 1)

# Scaling data
scaler = MinMaxScaler(copy=False)
scaler.fit(X)
scaler.transform(X)
MinMaxScaler(X, copy=False) # normalizes X in-place

# Making SVM
### Making the labels array (should have bought on day[i] if labels[i] >
y = np.array([1 if (opens[i] < opens[i + 1]) else 0 for i in range(0, len(opens) - 1)])

'''
Making correction, so that we can fit X to y directly (because y_i = f(X_(i-1)), and we
don't know the value of the stock tomorrow, we can only train y[1:] = f(X[:n-2]), where
n is the number of days/samples.
'''

total_days = 365*num_years
effective_prediction_days = total_days-2
y = y[1:] # can't make prediction from unknown X (on y_0, it would need X_-1, which is not available)
X = X[:effective_prediction_days]

train, cv, test = int(effective_prediction_days*.6), int(effective_prediction_days*.2), int(effective_prediction_days*.2)
train_data, train_labels = X[:train], y[:train]
cv_data, cv_labels = X[train:train+cv], y[train:train+cv]
test_data, test_labels = X[train+cv:], y[train+cv:]

print(X.shape)
print(y.shape)

print(X[:5])
print(y[:5])
print('opens:', opens[1:])