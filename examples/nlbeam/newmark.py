#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import io


from pyvib.newmark import newmark_beta_nl as newmark_beta
from pyvib.forcing import sineForce, randomPeriodic, sineSweep, toMDOF
from pyvib.nlforce import NL_force, NL_polynomial
from collections import namedtuple
import pickle


savedata = True

mat = io.loadmat('data/NLBeam.mat')
M = mat['M']
C = mat['C']
K = mat['K']

inl = np.array([[27,-1], [27,-1]])
enl = np.array([3,2])
knl = np.array([8e9,-1.05e7])
nl_pol = NL_polynomial(inl, enl, knl)
nl = NL_force(nl_pol)

# forcing parameters
ftype = 'sine'
#ftype = 'multisine'
ftype = 'sweep'
dt = 0.0002
fs = 1/dt

vrms = 3
f1 = 25
f2 = 40
vsweep = 10
fdof = 7
inctype = 'lin'
nper = 1
ndof = M.shape[0]

x0 = 0
xd0 = 0

# get external forcing
if ftype == 'multisine':
    nsper = 1
    u, t = randomPeriodic(vrms,fs, f1,f2,nsper, nper)
elif ftype == 'sweep':
    u, t, finst = sineSweep(vrms,fs, f1,f2,vsweep, nper, inctype)
    # we dont use the first value, as it is not repeated when the force is
    # generated. This is taken care of in randomPeriodic.
    nsper = (len(u)-1) // nper
elif ftype == 'sine':
    u, t = sineForce(vrms, f1, fs, nsper)
else:
    raise ValueError('Wrong type of forcing', ftype)
fext = toMDOF(u, ndof, fdof)
print('ns_tot     \t = %d' % len(u))

Nm = namedtuple('Nm', 'y yd ydd u t finst fs')

x, xd, xdd = newmark_beta(M, C, K, x0, xd0, dt, fext, nl, sensitivity=False)
sweep1 = Nm(x,xd,xdd,u,t,finst,fs)

u, t, finst = sineSweep(vrms,fs, f2,f1,-vsweep, nper, inctype)
fext = toMDOF(u, ndof, fdof)
x, xd, xdd = newmark_beta(M, C, K, x0, xd0, dt, fext, nl, sensitivity=False)
sweep2 = Nm(x,xd,xdd,u,t,finst,fs)

path = 'data/'
if savedata:
    pickle.dump(sweep1, open(path + 'sweep1.pkl', 'wb'))
    pickle.dump(sweep2, open(path + 'sweep2.pkl', 'wb'))
    print('data saved as {}'.format(path))


plt.figure()
plt.clf()
plt.plot(t,x[0],'-k')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (m)')
plt.title('Force type: {}, periods:{:d}'.format(ftype, nper))
plt.show()
