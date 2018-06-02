import numpy as np
from cryptocompy import price
#from cryptocompy import top

def top(limit):
    coin_data = top.get_top_coins('USD', limit=limit)
    coin_name = [p['SYMBOL'] for p in coin_data]
    coin_name

    return coin_name

def main():
    print(price.get_current_price("BTC", ["USD"]))
    data = price.get_historical_data('BTC', 'USD', 'minute', aggregate=1, limit=50)
    #print(price.get_historical_data('BTC', 'USD', 'minute', aggregate=5, limit=5))

    for idx in data:
        print ("Time: ", idx["time"], "Price: ", idx["close"])

    limit = 5
    #top(5)
    return 0

main()
