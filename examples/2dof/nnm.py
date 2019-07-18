#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle
import parameters

from pyvib.nnm import NNM
from pyvib.nlforce import NL_force, NL_polynomial

savedata = True

par = parameters.par

M = par['M']
K = par['K']

fdof = par['fdof']
vrms = par['vrms']
f1 = par['f1']
f2 = par['f2']

inl = par['inl']
enl = par['enl']
knl = par['knl']
nl_pol = NL_polynomial(inl, enl, knl)
nl = NL_force(nl_pol)

f2 = 2
par_nnm = {
    'omega_min': f1*2*np.pi,
    'omega_max': f2*2*np.pi,
    'opt_it_NR': 3,
    'max_it_NR': 15,
    'tol_NR': 1e-6,
    'step': 0.01,
    'step_min': 1e-6,
    'step_max': 1e-2,
    'scale': 1e-4,
    'angle_max_beta': 90,
    'adaptive_stepsize': True,
    'mode': 0,
    'unit': 'rad/s',
    'sca': 1
}

nnm1 = NNM(M, K, nl, **par_nnm)
nnm1.periodic()
nnm1.continuation()

par_nnm['mode'] = 1
nnm2 = NNM(M, K, nl, **par_nnm)
nnm2.periodic()
nnm2.continuation()

relpath = 'data/'
if savedata:
    pickle.dump(nnm1, open(relpath + 'nnm1' + '.pkl', 'wb'))
    pickle.dump(nnm2, open(relpath + 'nnm2' + '.pkl', 'wb'))
    print('data saved as {}'.format(relpath + 'nnm'))
