import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
import csv
from datetime import timedelta
from direction_svm import run_test

if __name__ == '__main__':
  run_test()