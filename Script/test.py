import alpaca_trade_api as tradeapi

api = tradeapi.REST(key_id='PKXVRTVRHQWTL50AYFKA',
                                 secret_key='',
                                 base_url='https://paper-api.alpaca.markets')
api.submit_order(
    symbol = "AAPL",
    qty=1,
    side='buy',
    type='market',
    time_in_force='gtc')
