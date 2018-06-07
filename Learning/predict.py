import numpy as np
from cryptocompy import price
from cryptocompy import top
import csv
from os import listdir
from os.path import isfile, join

"""
Issue: Our algorithm is outputting riducously high test values. This
leads to the question if the algorithm is accurate. If it is accurate, we need to
confirm and then there is little objective to continue training.

Objective: Because LSTM networks are based on time sequences, we can't pass in
independent values, we need to pass in the values with their associated time sequences.

Our algorithm needs to feed in the past day of {{Currency name}} prices and output
the prediction for one time series in the future. Aka predict for one minute in the future.

Pre Running this algorithm:
1. Save the parameters at epoch ~50 which contians very high test accuracy
2. Have algorithm able to predict for a single unit
#Predict the final prediction in the series

Break RNN's into RNN for training, and RNN for testing performance on a specific currency

We then need the algorithm to run every minute. Every minute the following occurs:

1. Append time vector
2. Pass time vector into RNN with great test accuracy
3. Predict change state for one minute in the future.
4. Record change state in array [[time, prediction, actual]]
5. Multiply change array by price array to get price change
6. Compare predicted change vs. market performance
"""


"""Seperately, I need to create the algorithm to concatinate data blocks of
currencies by time. To do this we need two algorithms:
1. Append data blocks by time and add them to each other correctly
2. Run every day"""


"""Model: Get everything working on multiple data inputs"""

"""
Add in a feature to save the hyperparameters set, and the loss value at different points.
Run with different sets of hyperparameters and optomize them for minimum test loss
"""


"""TO DO LIST:
Model:
1. Get model working on multiple inputs
2.

Data:
1. Finish preprocessing algorithms (merge alg)
2. Get tweet data sets
3. Get media data sets
4. Select the 100 currencies to analyze (store in list or something)
5. Concatinate data sets into singular csv dictionaries per currency per minute

Optimization:
1. Optomize that shit yo

Implementation:
1. Predict algorithm
2. Compare performance to market """


"""
1. Pick top 100 num_currencies
2. Scrape 2000 minute data
3. Concatinate data in time series, save to master data-set
4. Evolve model for multiple inputs


"""
