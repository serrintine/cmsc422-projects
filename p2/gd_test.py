from numpy import *
from pylab import *

import util
import imports
import datasets
import gd

# ----------------------------
# TESTING DIFFERENT STEP SIZES
# ----------------------------

x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, -1)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Gradient Descent with Step Size = -1')
show(True)

x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, -2)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Gradient Descent with Step Size = -2')
show(True)

x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 0)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Gradient Descent with Step Size = 0')
show(True)

x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 0.2)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Gradient Descent with Step Size = 0.2')
show(True)

x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 0.5)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Gradient Descent with Step Size = 0.5')
show(True)

x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 1)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Gradient Descent with Step Size = 1')
show(True)

x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 2)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Gradient Descent with Step Size = 2')
show(True)

x, trajectory = gd.gd(lambda x: x**2, lambda x: 2*x, 10, 100, 5)
plot(trajectory)
xlabel('Iteration Number')
ylabel('Approximated Value')
suptitle('Gradient Descent with Step Size = 5')
show(True)

# ---------------------------
# TESTING NON-CONVEX FUNCTION
# ---------------------------

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