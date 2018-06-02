import numpy as np
from cryptocompy import price
from cryptocompy import top



#Retrieve a vector of prices for the last day
def get_past_day_price():
    data = price.get_historical_data('BTC', 'USD', 'minute', aggregate=1, limit=(1440))
    price_vec = []
    for idx in data:
        price_vec.append(idx["close"])
        #print("Time: ", idx["time"], "Close: ", idx["close"])

    return price_vec

#Normalizes vector values to between zero and one
def normalize(vec):
    vec = np.diff(vec)
    vec[vec > 0] = 1
    vec[vec < 0] = 0
    """
    For non-binary training uncomment the following lines
    average = np.average(vec)
    vec = vec - average
    print(vec)
    max_val = np.amax(vec)
    vec = vec/max_val
    print(vec)"""

    return vec

print(normalize(get_past_day_price()))
