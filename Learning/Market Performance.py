import math
import numpy as np

def alg(days, npercent, ppercent, netmarket, avgchange):
    money = 100
    for i in range (int(days/2 * ppercent)):
        money = money * (1 + avgchange + (netmarket/days))
    for i in range (int(days/2 * (1 - npercent))):
        money = money * (1 - avgchange + (netmarket/days))

    return money

def market(days, npercent, ppercent, netmarket, avgchange):
    money = 100
    money = money*math.pow(float(1 + avgchange + (netmarket/days)), int(days/2))
    money = money*math.pow(float(1 - avgchange + (netmarket/days)), int(days/2))

    norm = -100*math.pow((1 + avgchange)*(1 - avgchange), int(days/2)) + 100

    return money + norm

def test():
    days = 1440*365
    npercent = 0.95
    ppercent = 0.95
    netmarket = 0.00
    avgchange = 0.0001
    print("\nMinutes: %d" % days)
    print("Negative Prediction Accuracy: %.2f" % (npercent * 100) + "%")
    print("Positive Prediction Accuracy: %.2f" % (ppercent * 100) + "%")
    print("Avg. Mag of Minute Market Change: %.2f" % (avgchange * 100) + "%")
    print("\nReturn Algorthm: %.2f" % (alg(days, npercent, ppercent, netmarket, avgchange) - 100) + "%")
    print("Return Markets: %.2f"  % (netmarket * 100) + "%")

    print("\nAlgorithm results are an underestimate")
test()
