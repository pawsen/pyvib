#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.linalg import norm

from pyvib.common import db
from pyvib.frf import covariance
from pyvib.pnlss import PNLSS
from pyvib.signal import Signal
from pyvib.subspace import Subspace

"""PNLSS model of the silverbox.

The Silverbox system can be seen as an electroninc implementation of the
Duffing oscilator. It is build as a 2nd order linear time-invariant system with
a 3rd degree polynomial static nonlinearity around it in feedback. This type of
dynamics are, for instance, often encountered in mechanical systems.
nonlinearity. The input-output data is synthetic.

See http://www.nonlinearbenchmark.org/#Silverbox
"""

# save figures to disk
savefig = True
savedata = True

data = sio.loadmat('data/SNLS80mV.mat')
# partitioning the data
u = data['V1'].T
y = data['V2'].T
u -= u.mean()
y -= y.mean()

R = 9
P = 1
fs = 1e7/2**14
m = 1         # number of inputs
p = 1         # number of outputs

npp = 8192
Nini = 86          # number of initial samples before the test starts.
Ntest = int(40e3)  # number of validation samples.
Nz = 100           # number of zero samples separating the blocks visually.
Ntr = 400          # number of transient samples.

# odd multisine till 200 Hz, without DC.
lines = np.arange(1,2683,2)

# partitioning the data
# Test data: only 86 zero samples in the initial arrow-like input.
utest = u[Nini     + np.r_[:Ntest+Ntr]]
ytest = y[Nini+Ntr + np.r_[:Ntest]]

# Estimation data.
u = np.delete(u, np.s_[:Nini+Ntr+Ntest])
y = np.delete(y, np.s_[:Nini+Ntr+Ntest])

uest = np.empty((npp,R,P))
yest = np.empty((npp,R,P))
for r in range(R):
    u = np.delete(u, np.s_[:Nz+Ntr])
    y = np.delete(y, np.s_[:Nz+Ntr])

    uest[:,r] = u[:npp,None]
    yest[:,r] = y[:npp,None]

    u = np.delete(u, np.s_[:npp])
    y = np.delete(y, np.s_[:npp])

#uest = np.repeat(uest[:,None],p,axis=1)
uest = uest.reshape(npp,m,R,P)
yest = yest.reshape(npp,p,R,P)
Pest = yest.shape[-1]

# Validation data.
u = np.delete(u, np.s_[:Nz+Ntr])
y = np.delete(y, np.s_[:Nz+Ntr])

uval = u[:npp, None]
yval = y[:npp, None]

# model orders and Subspace dimensioning parameter
n = 2
maxr = 20

sig = Signal(uest,yest,fs=fs)
sig.lines = lines
# estimate bla, total distortion, and noise distortion
sig.bla()
# average signal over periods. Used for training of PNLSS model
# Even if there's only 1 period, pnlss expect this to be run first. It also
# reshapes the signal, so um: (npp*m*R)
um, ym = sig.average()

linmodel = Subspace(sig)
# estimate bla, total distortion, and noise distortion
linmodel.estimate(n,maxr)
linmodel2 = deepcopy(linmodel)
linmodel2.optimize(weight=True)

# estimate PNLSS
# transient: Add two periods before the start of each realization. Note that
# this for the signal averaged over periods
T1 = np.r_[2*npp, np.r_[0:(R-1)*npp+1:npp]]

pnlss1 = PNLSS(linmodel2)
pnlss1.nlterms('x', [2,3], 'full')
# pnlss1.nlterms('y', [2,3], 'empty')
pnlss1.transient(T1)

covY = np.ones((round(npp//2),1,1))
pnlss2= deepcopy(pnlss1)
pnlss1.optimize(weight=False, nmax=50)
pnlss2.optimize(weight=covY, nmax=50)
models = [linmodel, linmodel2, pnlss1, pnlss2]
descrip = ('Subspace','Subspace opt','pnlss','pnlss weight')
nmodels = len(models)

# simulation error
est = np.empty((len(models),len(um)))
val = np.empty((len(models),len(uval)))
test = np.empty((len(models),len(utest)))

# add two transient period
Ptr2 = 2
opt_path = [[] for i in range(nmodels)]
for i, model in enumerate(models):
    est[i] = model.simulate(um, T1=Ptr2*npp)[1].T
    try:
        nl_errvec = model.extract_model(yval, uval, T1=Ptr2*npp)
        opt_path[i].append(nl_errvec)
    except:
        pass
    val[i] = model.simulate(uval, T1=Ptr2*npp)[1].T
    test[i] = model.simulate(utest, T1=0)[1].T

# delete transient data from test
test = np.delete(test, np.s_[:Ntr], axis=1)

rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
est_err = np.hstack((ym, (ym.T - est).T))
val_err = np.hstack((yval, (yval.T - val).T))
test_err = np.hstack((ytest, (ytest.T - test).T))
noise = np.abs(np.sqrt(Pest*covY.squeeze()))

print(descrip)
print(f'rms error est:\n    {rms(est_err[:,1:])}\ndb: {db(rms(est_err[:,1:]))}')
print(f'rms error val:\n    {rms(val_err[:,1:])}\ndb: {db(rms(val_err[:,1:]))}')
print(f'rms error test:\n    {rms(test_err[:,1:])}\ndb: {db(rms(test_err[:,1:]))}')

if savedata:
    data = {'models':models, 'opt_path':opt_path, 'est_err':est_err,
            'val_err':val_err, 'test_err':test_err, 'descrip':descrip}
    pickle.dump(data, open('sn_benchmark_pnlss.pkl', 'bw'))



## Plots ##
# store figure handle for saving the figures later
figs = {}

#plt.ion()
# result on estimation data
resamp = 20
plt.figure()
plt.plot(est_err[::resamp])
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(('Output',) + descrip)
plt.title('Estimation results')
figs['estimation_error'] = (plt.gcf(), plt.gca())

# result on validation data
resamp = 2
plt.figure()
plottime = val_err
N = plottime.shape[0]
freq = np.arange(N)/N*fs
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[1:nfd//2:resamp], db(plotfreq[1:nfd//2:resamp]), '.')
#plt.ylim([-110,10])
plt.xlim((0, 300))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

# result on test data
resamp = 2
plt.figure()
plottime = test_err
N = plottime.shape[0]
freq = np.arange(N)/N*fs
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[1:nfd//2:resamp], db(plotfreq[1:nfd//2:resamp]), '.')
#plt.ylim([-110,10])
plt.xlim((0, 300))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Test results')
figs['test_data'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
for err in [opt_path[2], opt_path[3]]:
    plt.plot(db(err))
    imin = np.argmin(err)
    plt.scatter(imin, db(err[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.legend(descrip[2:])
plt.title('Selection of the best model on a separate data set')
figs['fnsi_path'] = (plt.gcf(), plt.gca())

# subspace plots
figs['subspace_optim'] = linmodel.plot_info()
figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/SNbenchmark_{k}{i}.pdf")
