from numpy import *
from pylab import *

import util
import imports
import datasets
import gd

print gd.gd(lambda x: x**2, lambda x: 2*x, 10, 10, 0.2)
x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 0.2)
print x
x, trajectory = gd.gd(lambda x: linalg.norm(x)**2, lambda x: 2*x, array([10,5]), 100, 0.2)
print x