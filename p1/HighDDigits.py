from math import *
import random
from numpy import *
import matplotlib.pyplot as plt
import util
import datasets

waitForEnter=False

def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]

def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]

def computeExampleDistance(x1, x2):
    dist = 0.0
    for d in range(len(x1)):
        dist += (x1[d] - x2[d]) * (x1[d] - x2[d])
    return sqrt(dist)


def subsampleExampleDistance(x1, x2, D):
    dist = 0.0
    arr = []
    for x in range(0, 784):
        arr += [x]

    util.permute(arr)

    for d in range(D):
        dist += (x1[arr[d]] - x2[arr[d]]) * (x1[arr[d]] - x2[arr[d]])
    return sqrt(dist)

def computeDistancesDownsampled(data, d):
    dist = []
    print shape(data.X)
    for n in range(100):
        for m in range(n):
            dist.append( subsampleExampleDistance(data.X[n], data.X[m], d) / sqrt(d))
    return dist

Digits = [2, 8, 32, 128, 512]   # dimensionalities to try
Cols = ['#FF0000', '#880000', '#000000', '#000088', '#0000FF']
Bins = arange(0, 1, 0.02)

plt.xlabel('distance / sqrt(dimensionality)')
plt.ylabel('# of pairs of points at that distance')
plt.title('dimensionality versus uniform point distances')

for i,d in enumerate(Digits):
    distances = computeDistancesDownsampled(datasets.DigitData, d)
    print "D=%d, average distance=%g" % (d, mean(distances) * sqrt(d))
    plt.hist(distances,
             Bins,
             histtype='step',
             color=Cols[i])
    if waitForEnter:
        plt.legend(['%d dims' % d for d in Digits])
        plt.show(False)
        x = raw_input('Press enter to continue...')


plt.legend(['%d dims' % d for d in Digits])
plt.savefig('fig.pdf')
plt.show()
