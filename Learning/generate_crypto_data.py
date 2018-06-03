import numpy as np
from cryptocompy import price
from cryptocompy import top



#Retrieve a vector of prices for the last day
def get_past_day_price(currency):
    data = price.get_historical_data(currency, 'USD', 'minute', aggregate=1, limit=(1440))
    price_vec = []
    for idx in data:
        price_vec.append(idx["close"])
        #print("Time: ", idx["time"], "Close: ", idx["close"])

    return price_vec

#Retrieve top n currency's
def top_n(n):
    coins = top.get_top_coins('USD', limit = n)
    price_matrix = np.zeros((1441,1))
    count = 0
    for i in coins:
        count += 1
        #print(price_matrix.shape)
        price_matrix = np.c_[price_matrix, get_past_day_price(i["SYMBOL"])]
        #print(price_matrix)
        print(count)

    return price_matrix

#Normalizes vector values to between zero and one
def normalize(vec):
    vec = vec[:, 1:]
    vec = np.diff(vec, axis=0)
    max_val = np.amax(vec, axis=0)
    vec = vec/max_val
    """
    For non-binary training uncomment the following lines
    average = np.average(vec)
    vec = vec - average
    print(vec)
    max_val = np.amax(vec)
    vec = vec/max_val
    print(vec)"""

    return vec
