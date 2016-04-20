from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
from datasets import *

t = multiclass.makeBalancedTree(range(20))
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineData.X, WineData.Y)
P = h.predictAll(WineData.Xte)

print "Accuracy on full wine dataset"
print mean(P == WineData.Yte)