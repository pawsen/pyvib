#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import pickle
from matplotlib2tikz import save as tikz_save

savefig = True

abspath = ''
path = abspath + 'data/'
filename = path + 'stepped'
data_inc = np.load(filename + '_inc.npz')
data_dec = np.load(filename + '_dec.npz')
hb = pickle.load(open(path + 'hb.pkl', 'rb'))

def tohz(y):
    return y/(2*np.pi)

def steady_amp(data):
    # compute steady state amplitudes
    omegas = data['OMEGA_vec']
    y_steady = data['y_steady']
    A = []
    for i in range(len(omegas)):
        y1 = y_steady[i,0]
        y2 = y_steady[i,1]
        ymax = np.max(y1)
        ymin = np.min(y1)
        A.append(np.abs(0.5*(ymax-ymin)))
    return A

A_inc = steady_amp(data_inc)
A_dec = steady_amp(data_dec)

# plot amplitude against omega. Ie. FRF
fig1 = plt.figure(1)
plt.clf()
plt.plot(data_inc['OMEGA_vec'][::3], A_inc[::3], 'kx',
          label=r'$\Delta \Omega>0$')
plt.plot(data_dec['OMEGA_vec'][::3], A_dec[::3], 'ro', mfc='none',
          label=r'$\Delta \Omega<0$')

dof = 0
x = np.asarray(hb.omega_vec)/hb.scale_t
y = np.asarray(hb.xamp_vec).T[dof]
stab_vec = np.asarray(hb.stab_vec)
idx1 = ~stab_vec
idx2 = stab_vec
plt.plot(np.ma.masked_where(idx1, x),
         np.ma.masked_where(idx1, y), '-k',
         np.ma.masked_where(idx2, x),
         np.ma.masked_where(idx2, y), '--k')


plt.legend()
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Amplitude (m)')

fnfrc = plt.gcf()


def save(fig, filename):
    fig.savefig(filename + '.pdf')
    tikz_save(filename + '.tikz', figure=fig, show_info=False, strict=True)

if savefig:
    path = abspath + 'plots/'
    save(fnfrc, path + 'nfrc')

plt.show()



