#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle
import parameters

from pyvib.hb.hb import HB
from pyvib.hb.hbcommon import hb_signal
from pyvib.nlforce import NL_force, NL_polynomial
from pyvib.helper.plotting import nfrc

savedata = True
par = parameters.par
M = par['M']
C = par['C']
K = par['K']

fdof = 0
vrms = 2
f0 = 0.6/2/np.pi
f1 = 1e-4/2/np.pi
f2 = 5/2/np.pi

inl = par['inl']
enl = par['enl']
knl = par['knl']

par_hb ={
    'NH': 5,
    'npow2': 8,
    'nu': 1,
    'stability': True,
    'rcm_permute': False,
    'tol_NR': 1e-6,
    'max_it_NR': 15,
    'scale_x': 1,
    'scale_t': 1,
    'amp0':1e-4,
    'xstr':'rad/s',
    'sca':1
}
par_cont = {
    'omega_cont_min': f1*2*np.pi,
    'omega_cont_max': f2*2*np.pi,
    'cont_dir': 1,
    'opt_it_NR': 3,
    'step': 0.001,
    'step_min': 0.05,
    'step_max': 0.01,
    'angle_max_pred': 90,
    'it_cont_max': 1e6,
    'adaptive_stepsize': True,
    'detect':{'fold':True,'NS':True,'BP':True},
    'default_bp':True,
}

nl_pol = NL_polynomial(inl, enl, knl)
nl = NL_force(nl_pol)
hb = HB(M,C,K,nl, **par_hb)
hb.periodic(f0, vrms, fdof)
hb.continuation(**par_cont)

if savedata:
    relpath = 'data/'
    filename = relpath + 'hb.pkl'
    pickle.dump(hb, open(filename, "wb"))
    print('data saved as {}'.format(filename))

ffrf, ax = nfrc(dof=0, hb=hb, interactive=False, xscale=1, xunit='(rad/s)')
plt.show()
