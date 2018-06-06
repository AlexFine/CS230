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

We then need the algorithm to run every minute. Every minute the following occurs:

1. Append time vector
2. Pass time vector into RNN with great test accuracy
3. Predict change state for one minute in the future.
4. Record change state in array [[time, prediction, actual]]
5. Multiply change array by price array to get price change
6. Compare predicted change vs. market performance
"""
