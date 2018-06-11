# get data from  https://archive.ics.uci.edu/ml/datasets/iris


import pandas as pd
pd.read_csv("bezdekIris.data", header=None)


import math
import collections
def dist(a, b):
    sqSum = 0
    for i in range(len(a)):
        sqSum += (a[i] - b[i]) ** 2
    return math.sqrt(sqSum)

def kNN(k, train, given):
    distances = []
    for t in train:
        distances.append((dist(t[:-1], given), t[-1]))
    distances.sort()
    return distances[:k]

def kNN_classify(k, train, given):
    tally = collections.Counter()
    for nn in kNN(k, train, given):
        tally.update(nn[-1])
    return tally.most_common(1)[0]


