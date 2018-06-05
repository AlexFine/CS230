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

get_price()
