from numpy import *
from pylab import *

import util
import imports
import datasets
import gd

# -------------------------------
# WU4 TESTING NON-CONVEX FUNCTION
# -------------------------------

x, trajectory = gd.gd(lambda x: sin(3*x) + x**2, lambda x: 3*cos(3*x) + 2*x, 0, 100, 0.1)
print "Results for x0 = 0"
print gd.gd(lambda x: sin(3*x) + x**2, lambda x: 3*cos(3*x) + 2*x, 0, 100, 0.1)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Global Min of f(x) = sin(3*x) + x^2')
show(True)

print "\n--------------------------------------------------\n"

x, trajectory = gd.gd(lambda x: sin(3*x) + x**2, lambda x: 3*cos(3*x) + 2*x, 1, 100, 0.1)
print "Results for x0 = 1"
print gd.gd(lambda x: sin(3*x) + x**2, lambda x: 3*cos(3*x) + 2*x, 1, 100, 0.1)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Local Min of f(x) = sin(3*x) + x^2')
show(True)