#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import parameters
from pyvib.forcing import randomPeriodic

plot = True

def system_2dof(w, t, *args):
    #              x1                          x2                        #
    #            +-->                        +-->                        #
    #            |                           |                           #
    #  d1 __     +--------------+  d2 __     +--------------+  d3 __     #
    #-----__|----|              |-----__|----|              |-----__|----#
    #  k1        |              |  k3        |              |  k4        #
    #__/\  /\  __|      M1      |__/\  /\  __|       M2     |__/\  /\  __#
    #    \/  \/  |              |    \/  \/  |              |    \/  \/  #
    #  k2 ^      |              |            |              | k2 ^       #
    #__/\/ /\  __|              |            |              |__/\/ /\  __#
    #   /\/  \/  +--------------+            +--------------+   /\/  \/  #
    x1, x2, y1, y2 = w
    m1, m2, c1, c2, k1, k2, k3, mu1, mu2 = args

    global force
    # Create f = (x1',x2',y1',y2')
    f = [y1,
         y2,
         (-(k1 + k2) * x1 -c1*x1 + k2 * x2 - mu1 * x1**3 + force(t)) / m1,
         (k2 * x1 - (k2 + k3) * x2 -c2*x2 - mu2 * x2**3) / m2]
    return f

par = parameters.par
pars = [par[key] for key in ['m1', 'm2', 'c1', 'c2', 'k1', 'k2', 'k3', 'mu1', 'mu2']]
pars = tuple(pars)

fs = 20
ns = 1000
nrep = 2
f1 = 0.5/2/np.pi
f2 = 1.5/2/np.pi
vrms = 2.0
u,t = randomPeriodic(vrms, fs, f1, f2, ns, nrep=nrep)
force = interp1d(t, u, kind='linear')
t = t[:-1]

w0 = (0,0,0,0)
abserr = 1.0e-12
relerr = 1.0e-12
wsol = odeint(system_2dof, w0, t, args=pars, atol=abserr, rtol=relerr)

def recover_acc(t, y, v):
    """Recover the acceleration from the RHS:
    """
    a = np.empty(len(t))
    for i in range(len(t)):
        a[i] = system_2dof((y[i],v[i]),t[i],pars)[1]
    print('accelerations recovered')
    return a
   

plt.figure(1)
plt.clf()
plt.plot(t, wsol[:,0], '', label=r'$x_1$')
plt.plot(t, wsol[:,1], '--', label=r'$x_2$')
plt.legend(loc='best')
plt.xlabel('Time (t)')
plt.ylabel('Distance (m)')
plt.show()
