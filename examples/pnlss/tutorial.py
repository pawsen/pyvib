#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm

from pyvib.common import db
from pyvib.forcing import multisine
from pyvib.frf import covariance
from pyvib.pnlss import PNLSS
from pyvib.signal import Signal
from pyvib.subspace import Subspace

"""This tutorial shows the work flow of modeling a single input single output
(SISO) polynomial nonlinear state-space (PNLSS) model.

The tutorial shows how to
1) generate a multisine signal
2) compute the response of a SISO PNLSS model to the multisine input
3) estimate a nonparametric linear model from the input and noisy output data
4) estimate a parametric linear state-space model on the nonparametric model
5) estimate the parameters in the full PNLSS model

Translation of the tutorial provided with the matlab PNLSS program[1]_

[1]_
http://homepages.vub.ac.be/~ktiels/pnlss.html
"""

# save figures to disk
savefig = True
add_noise = True
weight = True

## Generate data from true model ##
# Construct model to estimate
A = np.array([[0.73915535, -0.62433133],[0.6247377, 0.7364469]])
B = np.array([[0.79287245], [-0.34515159]])
C = np.array([[0.71165154, 0.34917771]])
D = np.array([[0.04498052]])
E = np.array([[1.88130305e-01, -2.70291900e-01, 9.12423046e-03,
               -5.78088500e-01, 9.54588221e-03, 5.08576019e-04,
               -1.33890850e+00, -2.02171960e+00,-4.05918956e-01,
               -1.37744223e+00, 1.21206232e-01,-9.26349423e-02,
               -5.38072197e-01, 2.34134460e-03, 4.94334690e-02,
               -1.88329572e-02],
              [-5.35196110e-01, -3.66250013e-01, 2.34622651e-02,
               1.43228677e-01, -1.35959331e-02, 1.32052696e-02,
               7.98717915e-01, 1.35344901e+00, -5.29440815e-02,
               4.88513652e-01, 7.81285093e-01, -3.41019453e-01,
               2.27692972e-01, 7.70150211e-02, -1.25046731e-02,
               -1.62456154e-02]])
F = np.array([[-0.00867042, -0.00636662, 0.00197873, -0.00090865, -0.00088879,
               -0.02759694, -0.01817546, -0.10299409, 0.00648549, 0.08990175,
               0.21129849, 0.00030216, 0.03299013, 0.02058325, -0.09202439,
               -0.0380775]])

true_model = PNLSS(A, B, C, D, E, F)
true_model.nlterms('x', [2,3], 'full')
true_model.nlterms('y', [2,3], 'full')

# excitation signal
RMSu = 0.05   # Root mean square value for the input signal
npp = 1024    # Number of samples
R = 4         # Number of phase realizations (one for validation and one for
              # testing)
P = 3         # Number of periods
kind = 'Odd'  # 'Full','Odd','SpecialOdd', or 'RandomOdd': kind of multisine
m = 1         # number of inputs
p = 1         # number of outputs
fs = 1        # normalized sampling rate

# get predictable random numbers. https://dilbert.com/strip/2001-10-25
np.random.seed(10)
# shape of u from multisine: (R,P*npp)
u, lines, freq = multisine(N=npp, P=P, R=R, lines=kind, rms=RMSu)
# if multiple input is required, this will copy u m times

# Transient: Add one period before the start of each realization. To generate
# steady state data.
T1 = np.r_[npp, np.r_[0:(R-1)*P*npp+1:P*npp]]
_, y, _ = true_model.simulate(u.ravel(), T1=T1)
u = u.reshape((R,P,npp)).transpose((2,0,1))[:,None]  # (npp,m,R,P)
y = y.reshape((R,P,npp)).transpose((2,0,1))[:,None]

# Add colored noise to the output. randn generate white noise
if add_noise:
    np.random.seed(10)
    noise = 1e-3*np.std(y[:,-1,-1]) * np.random.randn(*y.shape)
    # Do some filtering to get colored noise
    noise[1:-2] += noise[2:-1]
    y += noise

## START of Identification ##
# partitioning the data. Use last period of two last realizations.
# test for performance testing and val for model selection
utest = u[:,:,-1,-1]
ytest = y[:,:,-1,-1]
uval = u[:,:,-2,-1]
yval = y[:,:,-2,-1]
# all other realizations are used for estimation
uest = u[...,:-2,:]
yest = y[...,:-2,:]
# noise estimate over periods. This sets the performace limit for the estimated
# model
covY = covariance(yest)
Pest = yest.shape[-1]

# create signal object
sig = Signal(uest,yest,fs=fs)
sig.lines = lines
# plot periodicity for one realization to verify data is steady state
# sig.periodicity()
# Calculate BLA, total- and noise distortion. Used for subspace identification
sig.bla()
# average signal over periods. Used for training of PNLSS model
um, ym = sig.average()

# model orders and Subspace dimensioning parameter
nvec = [2,3]
maxr = 5

linmodel = Subspace(sig)
# get best model on validation data
models, infodict = linmodel.scan(nvec, maxr, weight=weight)
l_errvec = linmodel.extract_model(yval, uval)
# or estimate the subspace model directly
linmodel.estimate(2, 5, weight=weight)  # best model, when noise weighting is used
linmodel.optimize(weight=weight)
print(f"Best subspace model, n, r: {linmodel.n}, {linmodel.r}")

# estimate PNLSS
# transient: Add one period before the start of each realization. Note that
# this is for the signal averaged over periods
Rest = yest.shape[2]
T1 = np.r_[npp, np.r_[0:(Rest-1)*npp+1:npp]]

model = PNLSS(linmodel)
model.nlterms('x', [2,3], 'full')
model.nlterms('y', [2,3], 'full')
model.transient(T1)
model.optimize(lamb=100, weight=weight, nmax=60)

# get best model on validation data. Change Transient settings, as there is
# only one realization
nl_errvec = model.extract_model(yval, uval, T1=npp)

models = [linmodel, model]
descrip = ('Linear', 'pnlss')  #, 'pnlss weight')
# simulation error
val = np.empty((len(models),len(uval)))
est = np.empty((len(models),len(um)))
test = np.empty((len(models),len(utest)))
for i, model in enumerate(models):
    test[i] = model.simulate(utest, T1=npp)[1].T
    val[i] = model.simulate(uval, T1=npp)[1].T
    est[i] = model.simulate(um, T1=T1)[1].T

rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
est_err = np.hstack((ym, (ym.T - est).T))
val_err = np.hstack((yval, (yval.T - val).T))
test_err = np.hstack((ytest, (ytest.T - test).T))
noise = np.abs(np.sqrt(Pest*covY.squeeze()))
print(f"err for models {descrip}")
print(f'rms error noise: {rms(noise)}\tdb: {db(rms(noise))} ')
print(f'rms error est:\n    {rms(est_err[:,1:])}\ndb: {db(rms(est_err[:,1:]))}')
print(f'rms error val:\n    {rms(val_err[:,1:])}\ndb: {db(rms(val_err[:,1:]))}')
print(f'rms error test:\n    {rms(test_err[:,1:])}\ndb: {db(rms(test_err[:,1:]))}')


## Plots ##
# store figure handle for saving the figures later
figs = {}

# linear and nonlinear model error
plt.figure()
plt.plot(est_err)
plt.xlabel('Time index')
plt.ylabel('Output (errors)')
plt.legend(('Output',) + descrip)
plt.title('Estimation results')
figs['estimation_error'] = (plt.gcf(), plt.gca())

# result on validation data
N = len(yval)
freq = np.arange(N)/N*fs
plottime = val_err
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.figure()
plt.plot(freq[lines], db(plotfreq[lines]), '.')
plt.plot(freq[lines], db(np.sqrt(Pest*covY[lines].squeeze() / N)), '.')
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Validation results')
figs['val_data'] = (plt.gcf(), plt.gca())

# result on test data
N = len(ytest)
freq = np.arange(N)/N*fs
plottime = test_err
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.figure()
plt.plot(freq[:nfd//2], db(plotfreq[:nfd//2]), '.')
plt.plot(freq[:nfd//2], db(np.sqrt(Pest*covY[:nfd//2].squeeze() / N)), '.')
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(('Output',) + descrip + ('Noise',))
plt.title('Test results')
figs['test_data'] = (plt.gcf(), plt.gca())

# optimization path for PNLSS
plt.figure()
plt.plot(db(nl_errvec))
imin = np.argmin(nl_errvec)
plt.scatter(imin, db(nl_errvec[imin]))
plt.xlabel('Successful iteration number')
plt.ylabel('Validation error [dB]')
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
            f[0].savefig(f"fig/tutorial_{k}{i}.pdf")

plt.show()
