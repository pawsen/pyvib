#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.linalg import norm

from pyvib.common import db
from pyvib.fnsi import FNSI
from pyvib.frf import covariance, periodic
from pyvib.helper.modal_plotting import plot_frf, plot_knl, plot_svg
from pyvib.modal import frf_mkc, modal_ac
from pyvib.signal import Signal
from pyvib.subspace import Subspace

"""FNSI model of the silverbox system.

The Silverbox system can be seen as an electroninc implementation of the
Duffing oscilator. It is build as a 2nd order linear time-invariant system with
a 3rd degree polynomial static nonlinearity around it in feedback. This type of
dynamics are, for instance, often encountered in mechanical systems.
nonlinearity. The input-output data is synthetic.
See http://www.nonlinearbenchmark.org/#Silverbox

This code correspond to the article
Grey-box state-space identification of nonlinear mechanical vibrations
JP. Noël & J. Schoukens
http://dx.doi.org/10.1080/00207179.2017.1308557

Values from paper:
Estimated nonliner coefficients at different sampling rates
| fs (Hz) |     c1 |   c2 |
|---------+--------+------|
|    2441 | -0.256 | 3.98 |
|   12205 | -0.267 | 3.96 |

Identified at low level (5V)
| Nat freq (Hz) | Damping ratio (%) |
|---------------+-------------------|
|         68.58 |              4.68 |

RMS of validation data
| Unit |   Output |  Lin err | fnsi init | fnsi opt |
|------+----------+----------+-----------+----------|
| V    |     0.16 |     0.09 |     0.002 |    0.001 |
| db   | -15.9176 | -20.9151 |  -53.9794 |     -60. |

"""

savedata = True

def load(var, amp, fnsi=True):
    fnsi = 'FNSI_' if fnsi else ''
    path = 'data/'
    fname = f"{path}SNJP_{var}m_full_{fnsi}{amp}.mat"
    data = sio.loadmat(fname)
    if var == 'u':
        um, fs, flines, P = [data[k] for k in ['um', 'fs', 'flines', 'P']]
        return um, fs.item(), flines.squeeze(), P.item()
    else:
        return data['ym']

# estimation data.
# 1 realization, 30 periods of 8192 samples. 5 discarded as transient (Ptr)
amp = 100
u, fs, lines, P = load('u',amp)
lines = lines - 1
y = load('y',amp)

NT, R = u.shape
NT, R = y.shape
npp = NT//P
Ptr = 5
m = 1
p = 1

# partitioning the data
u = u.reshape(npp,P,R,order='F').swapaxes(1,2)[:,None,:,Ptr:]
y = y.reshape(npp,P,R,order='F').swapaxes(1,2)[:,None,:,Ptr:]
# FNSI can only use one realization
uest = u[:,:,0,:]
yest = y[:,:,0,:]
Pest = yest.shape[-1]
# noise estimate over Pest periods
covY = covariance(yest[:,:,None])

# Validation data. 50 different realizations of 3 periods. Use the last
# realization and last period
uval_raw, _, _, Pval = load('u', 100, fnsi=False)
yval_raw = load('y', 100, fnsi=False)
uval_raw = uval_raw.reshape(npp,Pval,50,order='F').swapaxes(1,2)[:,None]
yval_raw = yval_raw.reshape(npp,Pval,50,order='F').swapaxes(1,2)[:,None]
uval = uval_raw[:,:,-1,-1]
yval = yval_raw[:,:,-1,-1]
utest = uval_raw[:,:,1,-1]
ytest = yval_raw[:,:,1,-1]
Rval = uval_raw.shape[2]

sig = Signal(uest,yest, fs=fs)
sig.lines = lines
um, ym = sig.average()
# sig.periodicity()

# for subspace model (from BLA)
sig2 = Signal(uest[:,:,None],yest[:,:,None], fs=fs)
sig2.lines = lines
sig2.bla()

# model orders and Subspace dimensioning parameter
n = 2
maxr = 20
dof = 0
iu = 0
xpowers = np.array([[2],[3]])

# subspace model
lin1 = Subspace(sig2)
# models, infodict = linmodel.scan(n, maxr, weight=False)
# ensure we use same dimension as for the fnsi model
lin1.estimate(n,maxr)
lin2 = deepcopy(lin1)
lin2.optimize(weight=False)

# Linear model
fnsi1 = FNSI(sig)
fnsi1.estimate(n,maxr)
fnsi1.nl_coeff(iu)

# initial nonlinear model
fnsi2 = FNSI(sig)
fnsi2.nlterms('state',xpowers)
fnsi2.estimate(n,maxr)
fnsi2.nl_coeff(iu)
fnsi2.transient(T1=npp)

covY = np.ones((round(npp//2),1,1))
# optimized models
fnsi3 = deepcopy(fnsi2)
fnsi4 = deepcopy(fnsi2)  # freq. weighted model
fnsi5 = deepcopy(fnsi2)  # freq. weighted model
weights = (False, True, covY)
for w, model in zip(weights,[fnsi3, fnsi4, fnsi5]):
    model.optimize(weight=w, nmax=50, xtol=1e-20, ftol=1e-20, gtol=1e-20)
    model.nl_coeff(iu)

models = [lin1, lin2, fnsi1, fnsi2, fnsi3, fnsi4, fnsi5]
descrip = ('subspace', 'subspace opt', 'fnsi linear','fnsi init',
           'fnsi opt', 'fnsi weight', 'fnsi unit')

# find validation error for all models
# add one transient period
Ptr2 = 2
nmodels = len(models)
opt_path = [[] for i in range(nmodels)]
knl = [[] for i in range(nmodels)]
x0 = [[] for i in range(nmodels)]
est = np.empty((nmodels,len(um)))
val_err = np.zeros((nmodels,Rval,len(uval)))
for i, model in enumerate(models):
    # get est error on best model
    est[i] = model.simulate(um, T1=npp)[1].T
    for j in range(Rval):
        uval = uval_raw[:,:,j,-1]
        yval = yval_raw[:,:,j,-1]
        try:
            # select best model on fresh data (val)
            nl_errvec = model.extract_model(yval, uval, T1=npp)
            opt_path[i].append(nl_errvec)
            knl[i].append(model.nl_coeff(iu)[1])
        except:
            pass
        x0[i].append(model.flatten())
        val = model.simulate(uval, T1=Ptr2*npp)[1].T
        val_err[i,j] = (yval.T - val)

rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
rms2 = lambda y: np.sqrt(np.mean(y**2, axis=2))
val_rms = rms2(val_err)
est_err = np.hstack((ym, (ym.T - est).T))
noise = np.abs(np.sqrt(Pest*covY.squeeze()))
print(descrip)
print(f'rms error noise. db: {db(rms(noise))} ')
print(f'rms error est:\n    {rms(est_err[:,1:])}\ndb: {db(rms(est_err[:,1:]))}')
print(f'rms error val:\n{val_rms.T}\ndb:\n{db(val_rms.T)}')
idx = np.argmin(val_rms,axis=1)
print(descrip)
print(f'Minimum db rms {db(val_rms.min(axis=1))}')
print(f'index {idx}')

if savedata:
    data = {'models':models, 'opt_path':opt_path, 'est_err':est_err,
            'val_err':val_err, 'descrip':descrip, 'knl': knl, 'x0':x0}
    pickle.dump(data, open('sn_jp_fnsi.pkl', 'bw'))
