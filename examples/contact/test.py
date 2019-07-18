import numpy as np
import matplotlib.pyplot as plt
from scipy import io

# from pyvib.interpolate import piecewise_linear

# delta = None
# delta = [5e-5,5e-5]
# slope = np.array([1e8, 0, 5e9])
# x = np.array([-1e-4, 5e-3])
# y = np.array([0,0])
# xx = np.linspace(-1e-2,1e-2,1000)
# yy = piecewise_linear(x,y,slope,delta=delta,xv=xx)
# yy2 = piecewise_linear(x,y,slope,delta=None,xv=xx)

# plt.ion()

# plt.figure(1)
# plt.clf()
# plt.plot(xx,yy)
# plt.plot(xx,yy2)
# plt.plot(x,y,'s')
# plt.show()

plt.ion()

beta = 1
alpha = 0
b = 0.1745
def f(y):
    f = y+(beta-alpha)*(np.abs(y-b) - np.abs(y+b))
    return f

x = np.linspace(-0.3, 0.3, 100)
y = f(x)
plt.figure(2)
plt.clf()
plt.plot(x,y)
# plt.show()
