from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from direction_svm import SVM_trader
import pandas as pd
import numpy as np

def main(universe):

  # Initializing variables (note there is no test data set, only cross validation)
  N, split, c_test_range, num_iterations = 10, (.8, .2, .0), \
                                           (1, 6000000), 10
  principle, can_short = 2000, False
  paths = [stock_name + '.csv' for stock_name in universe]

  # Making SVM for each stock in the universe
  S = SVM_trader(principle, can_short)
  for i, stock_name in enumerate(universe):
    # Updates class variable of best SVM for each stock in the universe
    data = np.array(pd.read_csv(paths[i], infer_datetime_format=True))

    print(data[-2])

    # S.make_svm(stock_name, N, paths[i], split, c_test_range, num_iterations)






if __name__ == '__main__':
  main(['AAPL'])