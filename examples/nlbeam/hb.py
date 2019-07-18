#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import io
import pickle
from collections import namedtuple

from pyvib.hb.hb import HB
from pyvib.hb.hbcommon import hb_signal
from pyvib.nlforce import NL_force, NL_polynomial
from pyvib.helper.plotting import (phase, periodic, stability, harmonic, nfrc)

savedata = True

mat = io.loadmat('data/NLBeam.mat')
M = mat['M']
C = mat['C']
K = mat['K']

# Force parameters
# location of harmonic force
fdofs = 7
f_amp = 3
# Excitation frequency. lowest sine freq in Hz
f0 = 25
par_hb ={
    'NH': 5,
    'npow2': 9,
    'nu': 1,
    'stability': True,
    'rcm_permute': False,
    'tol_NR': 1e-6,
    'max_it_NR': 15,
    'scale_x': 5e-6,  # == 5e-12
    'scale_t': 3000,
    'amp0': 1e-4
}

par_cont = {
    'omega_cont_min': 25*2*np.pi,
    'omega_cont_max': 40*2*np.pi,
    'cont_dir': 1,
    'opt_it_NR': 3,
    'step': 0.1,
    'step_min': 0.1,
    'step_max': 20,
    'angle_max_pred': 90,
    'it_cont_max': 1e4,
    'adaptive_stepsize': True
}


inl = np.array([[27,-1], [27,-1]])
enl = np.array([3,2])
knl = np.array([8e9,-1.05e7])
nl_pol = NL_polynomial(inl, enl, knl)
nl = NL_force()
nl.add(nl_pol)


hb = HB(M, C, K, nl, **par_hb)
omega, z, stab, B = hb.periodic(f0, f_amp, fdofs)
tp, omegap, zp, cnorm, c, cd, cdd = hb.get_components()
hb.continuation(**par_cont)

if savedata:
    filename = abspath + 'data/hb.pkl'
    pickle.dump(hb, open(filename, "wb"))
