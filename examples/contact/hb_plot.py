#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import pickle

from pyvib.hb.hbcommon import hb_components

savefig = True
# savefig = False

path='data/'
Nm = namedtuple('Nm', 'y yd ydd u t finst fs')
hb = pickle.load(open(path + 'hb.pkl', 'rb'))
sweep1 = pickle.load(open(path + 'pyds_sweepvrms0.03.pkl', 'rb'))

dof=0

plt.figure(1)
plt.clf()
plt.plot(sweep1.finst*2*np.pi, sweep1.y)
#plt.plot(sweep2.finst, sweep2.y[dof])
x = np.asarray(hb.omega_vec)/hb.scale_t
y = np.asarray(hb.xamp_vec).T[dof]
stab_vec = np.asarray(hb.stab_vec)
idx1 = ~stab_vec
idx2 = stab_vec
plt.plot(np.ma.masked_where(idx1, x),
         np.ma.masked_where(idx1, y), '-k',
         np.ma.masked_where(idx2, x),
         np.ma.masked_where(idx2, y), '--k')

# plt.plot(x,y,'-k')
plt.ylabel('Amplitude (m)')
plt.xlabel('Frequency (rad/s)')
fnfrc = plt.gcf()

# get harmonic components
NH = hb.NH
n = hb.n
cvec = []
for z in hb.z_vec:
    c, phi, cnorm = hb_components(z, n, NH)
    cvec.append(cnorm)
cvec = np.asarray(cvec)
cvec = np.moveaxis(cvec, 0, -1)
x = np.asarray(hb.omega_vec)/hb.scale_t

plt.figure(2)
plt.clf()
for i in range(cvec.shape[1]):
    plt.plot(x,cvec[dof,i])

plt.xlabel('Frequency (Hz)')
plt.ylabel('Nomalised components')
fhar = plt.gcf()

plt.show()

