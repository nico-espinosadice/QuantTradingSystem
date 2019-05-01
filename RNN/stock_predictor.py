from rnn import RNN

class Stock_Predictor:
	def __init__ (self, cash_available, stock_list):
		self.cash_available = cash_available
		self.stock_list = stock_list

	def predict_today(self):
		