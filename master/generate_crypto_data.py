import numpy as np
from cryptocompy import price
from cryptocompy import top
import csv
from os import listdir
from os.path import isfile, join

data_len = 2000
#Get list of 100 coins we're analyzing
def retrieve_coins():
    coins = []
    with open("coin_list.csv", newline='') as currency_file:
        #Get file data
        data = csv.reader(currency_file, delimiter=',', quotechar='|')
        #Loop through file data
        for row in data:
            #Append to output vector
            coins.append(str(row[0]))

    return coins

#Test reading from files
def read(dir):
    #Get a list of files from a directory
    currency_list = [f for f in listdir(dir) if isfile(join(dir, f))]
    return currency_list

#Return a vector of usable data from a directory
def read_data(dir):
    count = 0
    price_vec = []
    currency_matrix = np.zeros((data_len, 5))
    price_matrix = np.zeros((1,5))
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
                    for i in range(5):
                        price_vec.append(float(row[1+i]))
                count += 1

        currency_matrix = np.reshape(price_vec[0:data_len*5], (data_len, 5))
        price_matrix = np.append(price_matrix, currency_matrix, axis=0)
        price_vec = []
        count = 0

    price_matrix = price_matrix[1:, :]
    return price_matrix

#Return a vector of usable data for a single currency
def read_single_data(currency):
    count = 0
    price_vec = []
    price_matrix = np.zeros((1999,1))

    #Loop through each file name in directory
    with open(currency, newline='') as currency_file:
        #Get file data
        data = csv.reader(currency_file, delimiter=',', quotechar='|')
        #Loop through file data
        for row in data:
            #Append to output vector
            if count > 0:
                price_vec.append(float(row[1]))
            count += 1

    price_vec = price_vec[0:1999]
    count = 0

    return price_vec

#Retrieve a dictionary of vector of prices & time for the last day
def get_past_day_price(currency):
    data = price.get_historical_data(currency, 'USD', 'minute', aggregate=1, limit=(2001))
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
#Only needs to be run once
def list_top_n(n):
    coins = top.get_top_coins('USD', limit = n)
    download_dir = "coin_list.csv"
    csv = open(download_dir, "w")
    #csv.write("symbol" + "\n")

    for i in coins:
        print(i["SYMBOL"])
        csv.write(i["SYMBOL"])
        csv.write("\n")

    return 0

#Store top n currency's last day of data
def store_top_n(n):
    coins = retrieve_coins()

    #Iterate through list of coins
    count = 0
    for i in coins:
        count += 1
        print(count)
        #Currently we're normalizing the data before we store it
        download_dir = "normalized_2k_min_data/06-06/" + i + ".csv"
        #Open directory
        csv = open(download_dir, "w")
        #Write the header
        columnTitleRow = "time, price\n"
        csv.write(columnTitleRow)
        #Get the data to add
        data = get_past_day_price(i["SYMBOL"])
        price = normalize(data["price"]) #Normalize Prices
        time = data["time"]

        #Loop through time and price data
        for j in range(len(price)):
            row = time[j] + "," + str(price[j]) + "\n"
            csv.write(row)

    return 0

#Get the numeric difference over different time periods
def difference(arr, time):
    time = time
    temp = arr
    diff = arr[:-time]
    for i in range(time):
        diff = np.insert(diff, 0, 0)
    sub = np.subtract(arr, diff)
    sub[0:time] = 0

    return sub

#Normalizes vector va lues to between zero and one
def normalize_percent(vec, time):
    #vec = vec[1:]
    diff_vec = difference(vec, time)
    out = np.divide(diff_vec, vec, out=np.zeros_like(diff_vec), where=vec!=0) * 100
    #vec = diff_vec/vec[1:]
    #max_val = np.amax(vec)
    #vec = vec/max_val
    """
    For non-binary training uncomment the following lines
    average = np.average(vec)
    vec = vec - average
    print(vec)
    max_val = np.amax(vec)
    vec = vec/max_val
    print(vec)"""

    return out

#Transofrm vector into it's difference
def normalize_diff(vec):
    vec = vec[1:]
    diff_vec = np.diff(vec)
    #vec = diff_vec/vec[1:]
    #max_val = np.amax(vec)
    #vec = vec/max_val
    """
    For non-binary training uncomment the following lines
    average = np.average(vec)
    vec = vec - average
    print(vec)
    max_val = np.amax(vec)
    vec = vec/max_val
    print(vec)"""

    return diff_vec

#Store raw data for currency set
def store_raw(dir):
    coins = retrieve_coins()

    count = 0
    for i in coins:
        #i = [2:len(i)-2]
        count += 1
        print(count)
        download_dir = "data/" + dir + i + ".csv"
        csv = open(download_dir, "w")

        columnTitleRow = "time, price\n"
        csv.write(columnTitleRow)
        #Get the data to add
        data = get_past_day_price(i)
        price = data["price"] #Normalize Prices
        time = data["time"]

        #Loop through time and price data
        for j in range(len(price)):
            row = time[j] + "," + str(price[j]) + "\n"
            csv.write(row)

    return 0

#Store raw change for currency set
def store_raw_change(dir):
    coins = retrieve_coins()

    count = 0
    for i in coins:
        #i = [2:len(i)-2]
        count += 1
        print(count)
        download_dir = "data/" + dir + i + ".csv"
        csv = open(download_dir, "w")

        columnTitleRow = "time, price\n"
        csv.write(columnTitleRow)
        #Get the data to add
        data = get_past_day_price(i)
        price = normalize_diff(data["price"]) #Normalize Prices
        time = data["time"]

        #Loop through time and price data
        for j in range(len(price)):
            row = time[j] + "," + str(price[j]) + "\n"
            csv.write(row)

    return 0

#Store raw percent change for currency set
def store_raw_percent_change(dir):
    coins = retrieve_coins()

    count = 0
    for i in coins:
        #i = [2:len(i)-2]
        count += 1
        print(count)
        download_dir = "data/" + dir + i + ".csv"
        csv = open(download_dir, "w")

        columnTitleRow = "time, min1, min5, min15, min30, h1\n"
        csv.write(columnTitleRow)
        #Get the data to add
        data = get_past_day_price(i)
        min1 = normalize_percent(data["price"], 1) #Normalize Prices
        min5 = normalize_percent(data["price"], 5) #Normalize Prices
        min15 = normalize_percent(data["price"], 15) #Normalize Prices
        min30 = normalize_percent(data["price"], 30) #Normalize Prices
        h1 = normalize_percent(data["price"], 60) #Normalize Prices
        time = data["time"]

        #Loop through time and price data
        for j in range(len(min1)):
            row = time[j] + "," + str(min1[j]) + "," + str(min5[j]) + "," + str(min15[j]) + "," + str(min30[j]) + "," + str(h1[j]) + "\n"
            csv.write(row)

    return 0

#Merge two files by time period
def update(dir1, dir2):
    coins = retrieve_coins()

    for i in coins:
        #Download main directory to dictionary
        master_dict = csv_to_dictionary(dir1 + i)
        m_time_list = master_dict["time"]
        #Retrieve final time slot
        final_time = m_time_list[-1]

        #Download updated file
        append_dict = csv_to_dictionary(dir2 + i)
        a_time_list = append_dict["time"]
        #Find index of most recent time value
        index = a_time_list.index(final_time)
        print(index)

        ret_dict = {
            "time": append_dict["time"][index:],
            "min1": append_dict["min1"][index:],
            "min5": append_dict["min5"][index:],
            "min15": append_dict["min15"][index:],
            "min30": append_dict["min30"][index:],
            "h1": append_dict["h1"][index:]
        }
        print(dir1 + i)
        dictionary_to_csv(ret_dict, dir1 + i)

#Convert a csv to a dictionary
def csv_to_dictionary(dir):
    time = []
    min1 = []
    min5 = []
    min15 = []
    min30 = []
    h1 = []

    ret_file = {
        "time": time,
        "min1": min1,
        "min5": min5,
        "min15": min15,
        "min30": min30,
        "h1": h1
    }

    with open(dir + ".csv", newline='') as currency_file:
        #Get file data
        temp_data = csv.reader(currency_file, delimiter=',', quotechar='|')

        #Add to dictionary
        for row in temp_data:
            time.append(row[0])
            min1.append(row[1])
            min5.append(row[2])
            min15.append(row[3])
            min30.append(row[4])
            h1.append(row[5])

    return ret_file

#Convert a dictionary to a csv
def dictionary_to_csv(dict, dir):
    time_list = dict["time"]
    min1_list = dict["min1"]
    min5_list = dict["min5"]
    min15_list = dict["min15"]
    min30_list = dict["min30"]
    h1_list = dict["h1"]

    csv = open(dir + ".csv", "a")

    for i in range(len(time_list)):
        row = time_list[i] + ", " + min1_list[i] + ", " + min5_list[i] + ", " + min15_list[i] + ", " + min30_list[i] + ", " + h1_list[i] + "\n"
        csv.write(row)

    csv.close()

def master_update():
    store_raw_percent_change("temp_data/")
    update("data/normalized_price_data/", "data/temp_data/")

master_update()
