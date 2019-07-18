#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import pickle

from pyvib.nnm import NNM
from pyvib.nlforce import NL_force, NL_polynomial

savedata = True

mat = io.loadmat('data/NLBeam.mat')
M = mat['M']
K = mat['K']
inl = np.array([[27,-1], [27,-1]])
enl = np.array([3,2])
knl = np.array([8e9,-1.05e7])
nl_pol = NL_polynomial(inl, enl, knl)
nl = NL_force(nl_pol)

par_nnm = {
    'omega_min': 1*2*np.pi,
    'omega_max': 170*2*np.pi,
    'opt_it_NR': 3,
    'max_it_NR': 25,
    'tol_NR': 1e-6,
    'step': 0.001,
    'step_min': 1e-6,
    'step_max': 0.1,
    'scale': 1e-4,
    'angle_max_beta': 90,
    'adaptive_stepsize': True,
    'mode': 0,
    'anim':False
}

nnm1 = NNM(M, K, nl, **par_nnm)
nnm1.periodic()
nnm1.continuation()

par_nnm['mode'] = 1
nnm2 = NNM(M, K, nl, **par_nnm)
nnm2.periodic()
nnm2.continuation()

filename = 'data/nnm'
if savedata:
    pickle.dump(nnm1, open(filename + '1' + '.pkl', 'wb'))
    pickle.dump(nnm2, open(filename + '2' + '.pkl', 'wb'))

if nnm.anim:
    plt.show()
