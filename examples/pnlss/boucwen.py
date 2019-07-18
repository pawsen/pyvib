#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from copy import deepcopy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.linalg import norm

from pyvib.common import db
from pyvib.frf import covariance
from pyvib.pnlss import PNLSS
from pyvib.signal import Signal
from pyvib.subspace import Subspace

"""PNLSS model of BoucWen system with hysteresis acting as a dynamic
nonlinearity. The input-output data is synthetic.

From the paper by J.P. Noel: https://arxiv.org/pdf/1610.09138.pdf
| Polynomial degree | RMS validation error (dB) | Number of parameters |
|-------------------+---------------------------+----------------------|
|                 2 |                    -85.32 |                   34 |
|               2-3 |                    -90.35 |                   64 |
|             2-3-4 |                    -90.03 |                  109 |
|           2-3-4-5 |                    -94.87 |                  172 |
|         2-3-4-5-6 |                    -94.85 |                  256 |
|       2-3-4-5-6-7 |                    -97.96 |                  364 |
|             3-5-7 |                    -98.32 |                  217 |

See http://www.nonlinearbenchmark.org/#BoucWen
"""
def parse_cli(args):
    nlterms = {}
    for pair in args.nlterms:
        k, v = pair.split('=')
        if k not in nlterms:
            nlterms[k] = []
        nls = list(map(int, v.split(',')))
        nlterms[k].append(nls)
    return nlterms


# default values. Can be changed when running the script from CLI
savefig = True
savedata = True
weight = False  # Unit weight as specified in JP. article

# default. All models
nlterms = {'x':[[2], [2,3], [2,3,4], [2,3,4,5], [2,3,4,5,6], [2,3,4,5,6,7],
                [3,5,7]]}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", "-w", dest='weight', default=False,
                        action='store_true', help="set nonlinear weight")
    parser.add_argument("--nlterms", "-n", dest='nlterms',nargs='*',
                        help="set nonlinear weight")
    args = parser.parse_args()
    print(f'boucwen called with {args}')
    if args.nlterms is not None:
        nlterms = parse_cli(args)
    weight = args.weight


data = sio.loadmat('data/BoucWenData.mat')
# partition the data
uval = data['uval_multisine'].T
yval = data['yval_multisine'].T
utest = data['uval_sinesweep'].T
ytest = data['yval_sinesweep'].T
uest = data['u']
yest = data['y']
lines = data['lines'].squeeze()  # lines already are 0-indexed
fs = data['fs'].item()
nfreq = len(lines)
npp, m, R, P = uest.shape

# noise estimate over estimation periods
covY = covariance(yest)
Pest = yest.shape[-1]

# model orders and Subspace dimensioning parameter
nvec = [2,3,4]
maxr = 7

sig = Signal(uest,yest,fs=fs)
sig.lines = lines
sig.bla()
# average signal over periods. Used for estimation of PNLSS model
um, ym = sig.average()

linmodel = Subspace(sig)
models, infodict = linmodel.scan(nvec, maxr, nmax=50, weight=True)
# set model manual, as in matlab program
# linmodel.extract_model()
linmodel.estimate(n=3,r=4)
print(f'linear model: n,r:{linmodel.n},{linmodel.r}.')
print(f'Weighted Cost/nfreq: {linmodel.cost(weight=True)/nfreq}')

## estimate PNLSS ##
# transient: Add one period before the start of each realization of the
# averaged signal used for estimation
T1 = np.r_[npp, np.r_[0:(R-1)*npp+1:npp]]

# set common properties for PNLSS
pnlss = PNLSS(linmodel)
pnlss.transient(T1)

# initialize pnlss models specified by cli or default nlterms
models = [linmodel]
descrip = ['linmodel']
for k, value in nlterms.items():
    if not isinstance(value[0], list):
        value = [value]

    for v in value:
        model = deepcopy(pnlss)
        model.nlterms(k, v,'statesonly')
        models.append(model)
        # strip some unnecessary chars from the description
        # see more ways https://stackoverflow.com/q/3939361/1121523
        descrip.append(str(v).replace(" ","").replace("]","").replace("[",""))

opt_path = []
for desc, model in zip(descrip[1:], models[1:]):
    print(f"Optimizing xdegree: {str(model.xdegree)}")
    model.optimize(weight=weight, nmax=150)

    # get best model on validation data. Change Transient settings, as there is
    # only one realization
    nl_errvec = model.extract_model(yval, uval, T1=npp)
    opt_path.append(nl_errvec)
    if savedata:
        with open(f'boucwen_W_{weight}_model_{desc}.pkl', 'bw') as f:
            pickler = pickle.Pickler(f)
            data = {'linmodel': linmodel, 'model':model, 'errvec':nl_errvec,
                    'desc':desc}
            pickler.dump(data)

# add one transient period
Ptr2 = 1
# simulation error
# estimation simulation is done on final model, not model extracted on val data
est = np.empty((len(models),len(um)))
val = np.empty((len(models),len(uval)))
test = np.empty((len(models),len(utest)))
for i, model in enumerate(models):
    if isinstance(model, PNLSS):
        x0 = model.flatten()  # save best model on val data
        model._copy(*model.extract(model.res['x_mat'][-1]))  # best model on est
    est[i] = model.simulate(um, T1=T1)[1].T
    if isinstance(model, PNLSS):
        model._copy(*model.extract(x0))  # restore best model on val data
    val[i] = model.simulate(uval, T1=Ptr2*npp)[1].T
    test[i] = model.simulate(utest, T1=0)[1].T

rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
est_err = np.hstack((ym, (ym.T - est).T))
val_err = np.hstack((yval, (yval.T - val).T))
test_err = np.hstack((ytest, (ytest.T - test).T))
noise = np.abs(np.sqrt(Pest*covY.squeeze()))
print(f"err for models {descrip}")
print(f'rms error noise. db: {db(rms(noise))} ')
print(f'rms error est:\n    {rms(est_err[:,1:])}\ndb: {db(rms(est_err[:,1:]))}')
print(f'rms error val:\n    {rms(val_err[:,1:])}\ndb: {db(rms(val_err[:,1:]))}')
print(f'rms error test:\n    {rms(test_err[:,1:])}\ndb: {db(rms(test_err[:,1:]))}')

if savedata:
    fname = 'boucwen_lin.out'
    if len(models) > 1:
        fname = f'boucwen_W_{weight}_model_{descrip[1:]}.pkl'
    with open(fname, 'bw') as f:
        pickler = pickle.Pickler(f)
        data = {'linmodel':linmodel, 'models':models, 'opt_path':opt_path,
                'est_err':est_err, 'val_err':val_err, 'test_err':test_err,
                'descrip':descrip}
        pickler.dump(data)


## Plots ##
# store figure handle for saving the figures later
figs = {}

# plt.ion()
# linear and nonlinear model error
resamp = 20
plt.figure()
plt.plot(est_err[::resamp])
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(['Output'] + descrip)
plt.title('Estimation results')
figs['estimation_error'] = (plt.gcf(), plt.gca())

# result on validation data
resamp = 1
plt.figure()
plottime = val_err
N = plottime.shape[0]
freq = np.arange(N)/N*fs
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N**2)
nfd = plotfreq.shape[0]
plt.plot(freq[:nfd//2:resamp], db(plotfreq[:nfd//2:resamp]), '.')
plt.plot(freq[:nfd//2:resamp], db(noise[:nfd//2:resamp] / N), 'k.')
plt.xlim((5, 150))
plt.ylim((-160,-80))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(['Output'] + descrip + ['Noise'])
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

# result on test data
resamp = 30
plt.figure()
plottime = test_err
N = plottime.shape[0]
freq = np.arange(N)/N*fs
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[:nfd//2:resamp], db(plotfreq[:nfd//2:resamp]), '.')
plt.xlim((5, 200))
plt.ylim((-160,-45))
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(['Output'] + descrip + ['Noise'])
plt.title('Test results')
figs['test_data'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
for err in opt_path:
    plt.plot(db(err))
    imin = np.argmin(err)
    plt.scatter(imin, db(err[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
plt.legend(descrip[1:])
plt.title('Selection of the best model on a separate data set')
figs['pnlss_path'] = (plt.gcf(), plt.gca())

# subspace plots
figs['subspace_optim'] = linmodel.plot_info()
figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"boucwen_{k}{i}_W_{weight}_model_{descrip[1:]}.pdf")

plt.show()
