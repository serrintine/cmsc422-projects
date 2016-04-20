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