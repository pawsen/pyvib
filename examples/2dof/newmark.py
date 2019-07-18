#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigvals
import pickle
import parameters

from pyvib.nlforce import NL_force, NL_polynomial
from pyvib.forcing import sineForce, randomPeriodic, sineSweep, toMDOF
from pyvib.newmark import newmark_beta_nl as newmark_beta

plot = True
savedata = True
savefig = False

dof = 0

par = parameters.par

M = par['M']
C = par['C']
K = par['K']

ftype = 'sweep'
fdof = 0
vrms = 2
f1 = 0.001/2/np.pi
f2 = 5/np.pi
fs = 20*f2
nper = 2
nsper = 10000
vsweep = 0.01
inctype = 'lin'

inl = par['inl']
enl = par['enl']
knl = par['knl']
nl_pol = NL_polynomial(inl, enl, knl)
nl = NL_force(nl_pol)

x0, y0 = 0, 0
ndof = M.shape[0]
# get external forcing
if ftype == 'multisine':
    u, t = randomPeriodic(vrms, fs, f1, f2, nsper, nper)
elif ftype == 'sweep':
    u, t, finst = sineSweep(vrms, fs, f1, f2, vsweep, nper, inctype)
    # we dont use the first value, as it is not repeated when the force is
    # generated. This is taken care of in randomPeriodic.
    nsper = (len(u)-1) // nper
elif ftype == 'sine':
    u, t = sineForce(vrms, f=f1, fs=fs, ns=nsper)
else:
    raise ValueError('Wrong type of forcing', ftype)
fext = toMDOF(u, ndof, fdof)

dt = t[1] - t[0]
x, xd, xdd = newmark_beta(M, C, K, x0, y0, dt, fext, nl, sensitivity=False)


plt.figure()
plt.clf()
plt.plot(t, x[0], '-k', label=r'$x_1$')
plt.plot(t, x[1], '-r', label=r'$x_2$')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (m)')
plt.title('Force type: {}, periods:{:d}'.format(ftype, nper))
plt.legend()

plt.show()
