import numpy as np
from cryptocompy import price
from cryptocompy import top
import csv
from os import listdir
from os.path import isfile, join

def get_price():
    money = price.get_historical_data('BTC', 'USD', 'hour', aggregate=1, limit=(1000))
    print (len(money))
    print(money)
    return 0

#get_price()

def roll():
    a = np.reshape(np.arange(30), (10, 3))
    print(a)
    b = a[1:, :]
    b = b.astype(float)
    zero = np.zeros((1, len(a[1])))
    zero = zero.astype(float)
    print(zero)
    print(b)
    print(b.shape)
    print(zero.shape)
    b = np.vstack((b, zero))
    print(b)

    return 0

roll()
