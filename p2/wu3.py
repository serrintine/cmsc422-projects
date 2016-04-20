from numpy import *
from pylab import *

import util
import imports
import datasets
import gd

# --------------------------------
# WU3 TESTING DIFFERENT STEP SIZES
# --------------------------------

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