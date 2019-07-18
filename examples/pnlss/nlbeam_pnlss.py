#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyvib.statespace import StateSpace as linss
from pyvib.statespace import Signal
from pyvib.pnlss import PNLSS
from pyvib.common import db
from scipy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

"""PNLSS model of nonlinear beam

The NLBeam is synthetic data from a FEM simulation. It is made to resemble the
COST beam, ie. a clamped beam with a thin connection at the tip. The tip
connection can be seen as a nonlinear spring. """

# save figures to disk
savefig = True

data = sio.loadmat('data/NLbeam_u_15.mat')
lines = data['flines'].squeeze()  # TODO
fs = data['fs'].item()
npp = data['N'].item()
P = data['P'].item()
iu = data['iu'].item()
fmax = data['fmax'].item()
fmin = data['fmin'].item()
u = data['u']

datay = sio.loadmat('data/NLbeam_y_15.mat')
y = datay['y'].T

NT, m = u.shape
NT, p = y.shape
npp = NT//P
R = 1
# (npp,m,R,P)
u = u.reshape((npp,P,m,R)).transpose(0,2,3,1)
y = y.reshape((npp,P,p,R), order='F').transpose(0,2,3,1)

y = y[:,3:28:4]
p = 7

# partitioning the data
# validation data = last period of the last realization.
uval = u[:,:,-1,-1]
yval = y[:,:,-1,-1]

# estimation data
# discard first Ptr periods to get steady state data
Ptr = 3
uest = u[:,:,:R,Ptr:-1]
yest = y[:,:,:R,Ptr:-1]

# model orders and Subspace dimensioning parameter
n = 6
maxr = 7

sig = Signal(uest,yest,fs=1)
sig.lines(lines)
# um: (npp*R, m) # TODO is this the actual format of the output?
um, ym = sig.average()

linmodel = linss()
# estimate bla, total distortion, and noise distortion
linmodel.bla(sig)
models, infodict = linmodel.scan(n, maxr, weight=False)

# estimate PNLSS
# transient: Add three periods before the start of each realization. Note that
# this for the signal averaged over periods
T1 = np.r_[Ptr*npp, np.r_[0:(R-1)*npp+1:npp]]

model = PNLSS(linmodel.A, linmodel.B, linmodel.C, linmodel.D)
model.signal = sig
model.nlterms('x', [2,3], 'statesonly')
model.nlterms('y', [2,3], 'empty')
model.transient(T1)
model.optimize(weight=False, nmax=20)

# compute linear and nonlinear model output on training data
tlin, ylin, xlin = linmodel.simulate(um, T1=T1)
_, ynlin, _ = model.simulate(um)

# get best model on validation data. Change Transient settings, as there is
# only one realization
nl_errvec = model.extract_model(yval, uval, T1=Ptr*npp)

# compute model output on test data(unseen data)
_, ylval, _ = linmodel.simulate(uval, T1=Ptr*npp)
_, ynlval, _ = model.simulate(uval, T1=Ptr*npp)

# compute model output on test data(unseen data)
_, yltest, _ = linmodel.simulate(utest, T1=0)
_, ynltest, _ = model.simulate(utest, T1=0)
yltest = np.delete(yltest,np.s_[:Ntr])[:,None]
ynltest = np.delete(ynltest,np.s_[:Ntr])[:,None]

## Plots ##
# store figure handle for saving the figures later
figs = {}

plt.ion()
# linear and nonlinear model error
resamp = 1
plt.figure()
plt.plot(ym[::resamp])
plt.plot(ym[::resamp]-ylin[::resamp])
plt.plot(ym[::resamp]-ynlin[::resamp])
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(('output','linear error','PNLSS error'))
plt.title('Estimation results')
figs['estimation_error'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
plt.plot(db(nl_errvec))
imin = np.argmin(nl_errvec)
plt.scatter(imin, db(nl_errvec[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.title('Selection of the best model on a separate data set')
figs['pnlss_path'] = (plt.gcf(), plt.gca())

# result on validation data
plt.figure()
N = len(yval)
resamp = 1
freq = np.arange(N)/N*fs
# plot only for the DOF with the nonlinear connection
plottime = np.hstack((yval, yval-ylval, yval-ynlval))[:,6]
plotfreq = np.fft.fft(plottime, axis=0) / sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[1:nfd//2:resamp], db(plotfreq[1:nfd//2:resamp]), '.')
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output','Linear error','PNLSS error'))
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

# subspace plots
figs['subspace_optim'] = linmodel.plot_info()
figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"SNbenchmark_{k}{i}.pdf")
