import numpy as np
from cryptocompy import price
from cryptocompy import top
import csv
from os import listdir
from os.path import isfile, join

#Test reading from files
def read(dir):
    #Get a list of files from a directory
    currency_list = [f for f in listdir(dir) if isfile(join(dir, f))]
    return currency_list

#Return a vector of usable data
def read_data(dir):
    count = 0
    price_vec = []
    price_matrix = np.zeros((1999,1))
    currency_list = read(dir)

    #Loop through each file name in directory
    for currency in currency_list:
        #Open file
        with open(dir + currency, newline='') as currency_file:
            #Get file data
            data = csv.reader(currency_file, delimiter=',', quotechar='|')
            #Loop through file data
            for row in data:
                #Append to output vector
                if count > 0:
                    price_vec.append(float(row[1]))
                count += 1

        price_vec = price_vec[0:1999]
        price_matrix = np.c_[price_matrix, price_vec]
        price_vec = []
        count = 0

    return price_matrix

#Retrieve a dictionary of vector of prices & time for the last day
def get_past_day_price(currency):
    data = price.get_historical_data(currency, 'USD', 'minute', aggregate=1, limit=(1440))
    price_vec = []
    time_vec = []
    data_dictionary = {"time": time_vec, "price": price_vec}

    for idx in data:
        price_vec.append(idx["close"])
        time_vec.append(idx["time"])
        #print("Time: ", idx["time"], "Close: ", idx["close"])

    return data_dictionary

#Retrieve a dictionary of vector of prices & time for the last 500 day per hour
def get_past_hour_price(currency):
    data = price.get_historical_data(currency, 'USD', 'hour', aggregate=1, limit=(500*24))
    price_vec = []
    time_vec = []
    data_dictionary = {"time": time_vec, "price": price_vec}

    for idx in data:
        price_vec.append(idx["close"])
        time_vec.append(idx["time"])
        #print("Time: ", idx["time"], "Close: ", idx["close"])

    return data_dictionary

#Retrieve top n currency's
#DEPRICATED FUNCTION
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

#Store top n currency's last day of data
def store_top_n(n):
    coins = top.get_top_coins('USD', limit = n)

    #Iterate through list of coins
    count = 0
    for i in coins:
        count += 1
        print(count)
        #Currently we're normalizing the data before we store it
        download_dir = "normalized_data/" + i["SYMBOL"] + ".csv"
        #Open directory
        csv = open(download_dir, "w")
        #Write the header
        columnTitleRow = "time, price\n"
        csv.write(columnTitleRow)
        #Get the data to add
        data = get_past_hour_price(i["SYMBOL"])
        price = normalize(data["price"]) #Normalize Prices
        time = data["time"]

        #Loop through time and price data
        for j in range(len(price)):
            row = time[j] + "," + str(price[j]) + "\n"
            csv.write(row)

    return 0

#Normalizes vector values to between zero and one
def normalize(vec):
    vec = vec[1:]
    vec = np.diff(vec)
    max_val = np.amax(vec)
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
