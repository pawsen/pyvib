#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eigvals
from pyvib.helper.plotting import (phase, periodic, stability, configuration,
                                 nfrc)

plot = False

path = 'data/'
nnm = pickle.load(open(path + 'nnm1' + '.pkl', 'rb'))

def plots(t, x, xd, xxd, lamb, T, dof, inl, ptype='displ', idx='', savefig=False):
    fper, ax = periodic(t, x, dof=0, ls='-', c='k')
    fper, ax = periodic(t, x, dof=1, fig=fper, ax=ax, ls='--', c='k')
    fphase, ax = phase(x, xd, dof, c='k')
    fconf, ax = configuration(x, c='k')


    dof = 0
# get full periodic solution for last entry
T = 2*np.pi/nnm.omega_vec[-1]
x, xd, xdd, PhiT, dt = nnm.numsim(nnm.X0_vec[-1], T)
lamb = eigvals(PhiT)
ns = x.shape[1]
t = np.arange(ns)*dt
if plot:
    plots(t, x, xd, xdd, lamb, T, dof, inl, ptype='displ', idx='', savefig=savefig)


ffrf, ax = nfrc(nnm=nnm, interactive=False, xscale=1/2/np.pi, xunit='(Hz)')
ffep, ax = nfrc(nnm=nnm, interactive=False, xscale=1/2/np.pi,
                      xunit='(Hz)', energy_plot=True)
