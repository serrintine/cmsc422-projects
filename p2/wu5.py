import runClassifier
import mlGraphics
import datasets
import linear

from numpy import *
from pylab import *

import util
import imports

print "SquaredLoss"
f = linear.LinearClassifier({'lossFunction': linear.SquaredLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.WineDataBinary)
print f

print "\n----------------------------------------------------\n"
print "LogisticLoss"
f = linear.LinearClassifier({'lossFunction': linear.LogisticLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.WineDataBinary)
print f

print "\n----------------------------------------------------\n"
print "HingeLoss"
f = linear.LinearClassifier({'lossFunction': linear.HingeLoss(), 'lambda': 1, 'numIter': 100, 'stepSize': 0.5})
runClassifier.trainTestSet(f, datasets.WineDataBinary)
print f

print "\n----------------------------------------------------\n"
print "Word ID"
words = datasets.WineDataBinary().words
for i in range(0, len(words)):
	print str(i) + words[i]