#from matplotlib2tikz import save as tikz_save
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eigvals
from pyvib.helper.plotting import (phase, periodic, stability, configuration,
                                 nfrc)

filename = 'data/nnm'
nnm1 = pickle.load(open(filename + '1' + '.pkl', 'rb'))
nnm2 = pickle.load(open(filename + '2' + '.pkl', 'rb'))


fig1, ax1 = plt.subplots()
nfrc(nnm=nnm1, interactive=False, xscale=1, xunit='(rad/s)', fig=fig1, ax=ax1)
nfrc(nnm=nnm2, interactive=False, xscale=1, xunit='(rad/s)', fig=fig1, ax=ax1)

fig2, ax2 = plt.subplots()
nfrc(nnm=nnm1, interactive=False, xscale=1, xunit='(rad/s)', energy_plot=True,
     fig=fig2, ax=ax2)
nfrc(nnm=nnm2, interactive=False, xscale=1, xunit='(rad/s)', energy_plot=True,
     fig=fig2, ax=ax2)


plotlist = [periodic, phase, stability, configuration]
nfrc(nnm=nnm1, interactive=True, xscale=1, xunit='(rad/s)', energy_plot=True,
     plotlist=plotlist)

plt.show()

# # get full periodic solution
# T = 2*np.pi/nnm.omega_vec[-1]
# x, xd, xdd, PhiT, dt = nnm.numsim(nnm.X0_vec[-1], T)
# lamb = eigvals(PhiT)
# ns = x.shape[1]
# t = np.arange(ns)*dt
