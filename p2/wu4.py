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

x = linspace(-5, 5, 100)
y = sin(3*x) + x**2
plot(x, y)
plot(-0.4273, -0.7760, 'ro')
plot(1.2446, 0.9908, 'ro')
suptitle('Local and global mins of f(x) = sin(3*x) + x^2')
show(True)