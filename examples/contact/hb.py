#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pickle
import parameters

from pyvib.hb.hb import HB
from pyvib.hb.hbcommon import hb_signal
from pyvib.nlforce import NL_force, NL_piecewise_linear
from pyvib.helper.plotting import (phase, periodic, stability, harmonic,
                                 nfrc)


savedata = True

M = np.array([[1]])
C = np.array([[0.025*2]])
K = np.array([[1]])

fdof = 0
vrms = 0.03
f0 = 1e-3
f1 = 1e-4
f2 = 1.8
b = 0.1745
alpha = 0
beta = 3

inl = np.array([[0, -1]])
delta = [5e-4, 5e-4]
slope = np.array([beta, alpha, beta])
x = np.array([-b, b])
y = np.array([0, 0])

nl_piece = NL_piecewise_linear(x, y, slope, inl, delta)
nl = NL_force()
nl.add(nl_piece)

par_hb ={
    'NH': 12,
    'npow2': 8,
    'nu': 1,
    'stability': True,
    'rcm_permute': False,
    'tol_NR': 1e-6,
    'max_it_NR': 25,
    'scale_x': 1,
    'scale_t': 1,
    'amp0':1e-4,
    'xstr':'rad/s',
    'xscale':1
}
par_cont = {
    'omega_cont_min': f1,
    'omega_cont_max': f2,
    'cont_dir': 1,
    'opt_it_NR': 3,
    'step': 0.001,
    'step_min': 0.00001,
    'step_max': 0.01,
    'angle_max_pred': 90,
    'it_cont_max': 1e6,
    'adaptive_stepsize': True
}

# run full continuation
hb = HB(M,C,K,nl, **par_hb)
omega, z, stab, lamb = hb.periodic(f0, vrms, fdof)
hb.continuation(**par_cont)


if savedata:
    with open('data/hb' + '.pkl', 'wb') as f:
        pickle.dump(hb, f)
    print('data saved as {}'.format(filename))


ffrf, ax = nfrc(dof=0, hb=hb, interactive=False, xscale=1,
                      xunit='(rad/s)')


plt.show()

"""
One-sided:
    'NH': 12,
    'npow2': 8,
    'step_max': 0.01,
vrms = 0.01
b = 0.08
beta = 1
slope = np.array([0, -beta])
x = np.array([b])
y = np.array([0])

"""
