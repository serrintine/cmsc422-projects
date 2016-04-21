import runClassifier
import mlGraphics
import datasets
import linear

from numpy import *
from pylab import *

import util
import imports

print "SquaredLoss"
s = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(s, datasets.WineDataBinary)

print "\n----------------------------------------------------\n"
print "LogisticLoss"
l = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(l, datasets.WineDataBinary)

print "\n----------------------------------------------------\n"
print "HingeLoss"
h = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(h, datasets.WineDataBinary)

print "\n----------------------------------------------------\n"
print "HingeLoss weights + words"
words = datasets.WineDataBinary().words
wt, wd = (list(x) for x in zip(*sorted(zip(l.getRepresentation(), words))))
print "\nTop 5 negative weights + words"
print wt[:5]
print wd[:5]
print "\nTop 5 positive weights + words"
print wt[-5:]
print wd[-5:]