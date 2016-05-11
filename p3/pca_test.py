from numpy import *
from matplotlib.pyplot import *
import util
import dr
import datasets

(X,Y) = datasets.loadDigits()
(P,Z,evals) = dr.pca(X, 784)
print evals