import numpy as np

def test():
    a = np.arange(20)
    print(a)
    print(len(a))
    temp = a
    a = np.insert(a, 0, 0)
    c = a[1:] - temp
    print(np.diff(a))
    print(len(np.diff(a)))
    print(c)
    difference(a)

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

def reshape():
    x = np.arange(100).reshape(20,5)
    z = np.arange(20)*5
    size = 2
    print(x)
    y = x.reshape(size, -1, 5)
    z = z.reshape(size, -1)
    print(y.shape)
    print(z.shape)
    print(y)
    print(z)
