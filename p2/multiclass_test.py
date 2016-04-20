from sklearn.tree import DecisionTreeClassifier
import multiclass
import util
from datasets import *

t = multiclass.makeBalancedTree(range(6))
print t
print t.isLeaf
print t.getLeft()
print t.getLeft().getLeft()
print t.getLeft().getLeft().isLeaf

t = multiclass.makeBalancedTree(range(5))
h = multiclass.MCTree(t, lambda: DecisionTreeClassifier(max_depth=3))
h.train(WineDataSmall.X, WineDataSmall.Y)
P = h.predictAll(WineDataSmall.Xte)
print mean(P == WineDataSmall.Yte)