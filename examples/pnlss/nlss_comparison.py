#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from collections import namedtuple

from pyvib.common import db
from pyvib.forcing import multisine
from pyvib.frf import covariance
from pyvib.nlss import NLSS
from pyvib.fnsi import FNSI
from pyvib.nonlinear_elements import (Pnl, Tanhdryfriction, Nonlinear_Element)
from pyvib.signal import Signal
from pyvib.subspace import Subspace

"""This script shows:
- LM is not able to correctly estimate the system for many parameters.
- If the BLA is far away from the nl system, NLSS willl not converge(fnsi will)
- regulized tanh can identify sign-nonlinearity

This script works with git id: 71c612c
"""
# data containers
Data = namedtuple('Data', ['sig', 'uest', 'yest', 'uval', 'yval', 'utest',
                           'ytest', 'um', 'ym', 'covY', 'freq', 'lines',
                           'npp', 'Ntr'])
Result = namedtuple('Result', ['est_err', 'val_err', 'test_err', 'noise',
                               'nl_errvec', 'descrip'])

savefig = False
add_noise = False
weight = False
p = 2  # number of outputs

## Generate data from true model ##
# Construct model to estimate
A = np.array([[0.73915535, -0.62433133],[0.6247377, 0.7364469]])
B = np.array([[0.79287245], [-0.34515159]])
C = np.array([[0.71165154, 0.34917771]])
D = np.array([[0.04498052]])
if p == 2:
    C = np.vstack((C,C))
    D = np.vstack((D,0.1563532))

Ffull = np.array([
        [-0.00867042, -0.00636662,  0.00197873, -0.00090865, -0.00088879,
         -0.02759694, -0.01817546, -0.10299409,  0.00648549,  0.08990175,
          0.21129849,  0.00030216,  0.03299013,  0.02058325, -0.09202439,
         -0.0380775],
        [-0.17323214, -0.08738017, -0.11346953, -0.08077963, -0.05496476,
          0.01874564, -0.02946581, -0.01869213, -0.07492472,  0.06868484,
         -0.02770704,  0.19900055, -0.089364  , -0.00410125,  0.13002691,
         -0.11460958]])
Efull = np.array([
        [ 1.88130305e-01, -2.70291900e-01,  9.12423046e-03, -5.78088500e-01,
          9.54588221e-03,  5.08576019e-04, -1.33890850e+00, -2.02171960e+00,
         -4.05918956e-01, -1.37744223e+00,  1.21206232e-01, -9.26349423e-02,
         -5.38072197e-01,  2.34134460e-03,  4.94334690e-02, -1.88329572e-02],
        [-5.35196110e-01, -3.66250013e-01,  2.34622651e-02,  1.43228677e-01,
         -1.35959331e-02,  1.32052696e-02,  7.98717915e-01,  1.35344901e+00,
         -5.29440815e-02,  4.88513652e-01,  7.81285093e-01, -3.41019453e-01,
          2.27692972e-01,  7.70150211e-02, -1.25046731e-02, -1.62456154e-02]])

# excitation signal
RMSu = 0.05     # Root mean square value for the input signal
R = 4           # Number of phase realizations (one for validation and one
                # for testing)
P = 3           # Number of periods
kind = 'Odd' # 'Full','Odd','SpecialOdd', or 'RandomOdd': kind of multisine
m = D.shape[1]  # number of inputs
p = C.shape[0]  # number of outputs
fs = 1          # normalized sampling rate
def simulate(true_model,npp=1024, Ntr=1, Rest=2, add_noise=False):
    print()
    print(f'Nonlinear parameters:',
          f'{len(true_model.nlx.active) + len(true_model.nly.active)}')
    print(f'Parameters to estimate: {true_model.npar}')
    # set non-active coefficients to zero. Note order of input matters
    idx = np.setdiff1d(np.arange(true_model.E.size), true_model.nlx.active)
    idy = np.setdiff1d(np.arange(true_model.F.size), true_model.nly.active)
    true_model.E.flat[idx] = 0
    true_model.F.flat[idy] = 0

    # get predictable random numbers. https://dilbert.com/strip/2001-10-25
    np.random.seed(10)
    # shape of u from multisine: (R,P*npp)
    u, lines, freq, t = multisine(N=npp, P=P, R=R, lines=kind, rms=RMSu)
    
    # Transient: Add Ntr periods before the start of each realization. To
    # generate steady state data.
    T1 = np.r_[npp*Ntr, np.r_[0:(R-1)*P*npp+1:P*npp]]
    _, yorig, _ = true_model.simulate(u.ravel(), T1=T1)
    u = u.reshape((R,P,npp)).transpose((2,0,1))[:,None]  # (npp,m,R,P)
    y = yorig.reshape((R,P,npp,p)).transpose((2,3,0,1))

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
    uest = u[...,:Rest,:]
    yest = y[...,:Rest,:]
    # noise estimate over periods. This sets the performace limit for the
    # estimated model
    covY = covariance(yest)
    
    # create signal object
    sig = Signal(uest,yest,fs=fs)
    sig.lines = lines
    # plot periodicity for one realization to verify data is steady state
    # sig.periodicity()
    # Calculate BLA, total- and noise distortion. Used for subspace 
    # identification
    sig.bla()
    # average signal over periods. Used for training of PNLSS model
    um, ym = sig.average()

    return Data(sig, uest, yest, uval, yval, utest, ytest, um, ym, covY,
                freq, lines, npp, Ntr)
    
def identify(data, nlx, nly, nmax=25, info=2, fnsi=False):
    # transient: Add one period before the start of each realization. Note that
    # this is for the signal averaged over periods
    Rest = data.yest.shape[2]
    T1 = np.r_[data.npp*data.Ntr, np.r_[0:(Rest-1)*data.npp+1:data.npp]]   

    linmodel = Subspace(data.sig)
    linmodel._cost_normalize = 1
    linmodel.estimate(2, 5, weight=weight)
    linmodel.optimize(weight=weight, info=info)
    
    # estimate NLSS       
    model = NLSS(linmodel)
    # model._cost_normalize = 1
    model.add_nl(nlx=nlx, nly=nly)
    model.set_signal(data.sig)
    model.transient(T1)
    model.optimize(lamb=100, weight=weight, nmax=nmax, info=info)
    # get best model on validation data. Change Transient settings, as there is
    # only one realization
    nl_errvec = model.extract_model(data.yval, data.uval, T1=data.npp*data.Ntr,
                                    info=info)
    models = [linmodel, model]
    descrip = [type(mod).__name__ for mod in models]

    if fnsi:
        # FNSI can only use 1 realization
        sig = deepcopy(data.sig)
        # This is stupid, but unfortunately nessecary
        sig.y = sig.y[:,:,0][:,:,None]
        sig.u = sig.u[:,:,0][:,:,None]
        sig.R = 1
        sig.average()
        fnsi1 = FNSI()
        fnsi1.set_signal(sig)
        fnsi1.add_nl(nlx=nlx)
        fnsi1.estimate(n=2, r=5, weight=weight)
        fnsi1.transient(T1)
        fnsi2 = deepcopy(fnsi1)
        fnsi2.optimize(lamb=100, weight=weight, nmax=nmax, info=info)
        models = models + [fnsi1, fnsi2]
        descrip = descrip + ['FNSI', 'FNSI optimized']

    descrip = tuple(descrip)  # convert to tuple for legend concatenation
    # simulation error
    val = np.empty((*data.yval.shape, len(models)))
    est = np.empty((*data.ym.shape, len(models)))
    test = np.empty((*data.ytest.shape, len(models)))
    for i, model in enumerate(models):
        test[...,i] = model.simulate(data.utest, T1=data.npp*data.Ntr)[1]
        val[...,i] = model.simulate(data.uval, T1=data.npp*data.Ntr)[1]
        est[...,i] = model.simulate(data.um, T1=T1)[1]

    Pest = data.yest.shape[3]
    # convenience inline functions
    stack = lambda ydata, ymodel: \
        np.concatenate((ydata[...,None], (ydata[...,None] - ymodel)),axis=2)
    rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
    est_err = stack(data.ym, est)  # (npp*R,p,nmodels)
    val_err = stack(data.yval, val)
    test_err = stack(data.ytest, test)
    noise = np.abs(np.sqrt(Pest*data.covY.squeeze()))
    print()
    print(f"err for models: signal, {descrip}")
    # print(f'rms error noise:\n{rms(noise)}     \ndb: \n{db(rms(noise))} ')
    # only print error for p = 0. Almost equal to p = 1
    print(f'rms error est (db): \n{db(rms(est_err[:,0]))}')
    print(f'rms error val (db): \n{db(rms(val_err[:,0]))}')
    # print(f'rms error test: \n{rms(test_err)}  \ndb: \n{db(rms(test_err))}')
    return Result(est_err, val_err, test_err, noise, nl_errvec, descrip)

def plot(res, data, p):
    figs = {}
    lines = data.lines
    freq = data.freq
    Pest = data.yest.shape[3]

    # result on validation data
    N = len(data.yval)
    freq = np.arange(N)/N*fs
    plottime = res.val_err
    plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
    plt.figure()
    plt.plot(freq[lines], db(plotfreq[lines,p]), '.')
    plt.plot(freq[lines], db(np.sqrt(Pest*data.covY[lines,p,p].squeeze() / N)),
             '.')
    plt.xlabel('Frequency')
    plt.ylabel('Output (errors) (dB)')
    plt.legend(('Output',) + res.descrip + ('Noise',))
    plt.title(f'Validation results p:{p}')
    figs['val_data'] = (plt.gcf(), plt.gca())

    # optimization path for NLSS
    plt.figure()
    plt.plot(db(res.nl_errvec))
    imin = np.argmin(res.nl_errvec)
    plt.scatter(imin, db(res.nl_errvec[imin]))
    plt.xlabel('Successful iteration number')
    plt.ylabel('Validation error [dB]')
    plt.title('Selection of the best model on a separate data set')
    figs['pnlss_path'] = (plt.gcf(), plt.gca())
    
    return figs

def plot_time(res, data, p):
    figs = {}
    plt.figure()
    plt.plot(res.est_err[:,p])
    plt.xlabel('Time index')
    plt.ylabel('Output (errors)')
    plt.legend(('Output',) + res.descrip)
    plt.title(f'Estimation results p:{p}')
    figs['estimation_error'] = (plt.gcf(), plt.gca())
    return figs

def plot_bla(res, data, p):
    figs = {}
    lines = data.lines
    freq = data.freq

    # BLA plot. We can estimate nonlinear distortion
    # total and noise distortion averaged over P periods and M realizations
    # total distortion level includes nonlinear and noise distortion
    plt.figure()
    # When comparing distortion(variance, proportional to power) with 
    # G(propertional to amplitude(field)), there is two definations for dB:
    # dB for power: Lp = 10 log10(P). 
    # dB for field quantity: Lf = 10 log10(F²)
    # Alternative calc: bla_noise = db(np.abs(sig.covGn[:,pp,pp])*R, 'power')
    # if the signal is noise-free, fix noise so we see it in plot
    bla_noise = db(np.sqrt(np.abs(data.sig.covGn[:,p,p])*R))
    bla_noise[bla_noise < -150] = -150
    bla_tot = db(np.sqrt(np.abs(data.sig.covG[:,p,p])*R))
    bla_tot[bla_tot < -150] = -150

    plt.plot(freq[lines], db(np.abs(data.sig.G[:,p,0])))
    plt.plot(freq[lines], bla_noise,'s')
    plt.plot(freq[lines], bla_tot,'*')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('magnitude (dB)')
    plt.title(f'Estimated BLA and nonlinear distortion p: {p}')
    plt.legend(('BLA FRF','Noise Distortion','Total Distortion'))
    plt.gca().set_ylim(bottom=-150) 
    figs['bla'] = (plt.gcf(), plt.gca())
    return figs

nmax = 100
info = 1


"""Check optimization; how good is it to find the true system parameters?
This depends on the number of parameters, as we expect for a nonlinear
optimization problem."""

# Few parameters, LM able to estimate system properly
# Diagonal is only for state equation. If n == p, we can use diagonal for
# output, but that is not the intended usage.
E = Efull
F = Ffull[:p]
nlx = [Pnl(degree=[2,3], structure='diagonal')]
nly = [Pnl(degree=[2,3], structure='statesonly')]
true_model = NLSS(A, B, C, D, E, F)
true_model.add_nl(nlx=nlx, nly=nly)
data1 = simulate(true_model)       # generate data from true model
res1 = identify(data1, nlx, nly, nmax=nmax, info=info)    # estimate model from data

## Many parameters
nlx = [Pnl(degree=[2,3], structure='full')]
nly = [Pnl(degree=[2,3], structure='full')]
true_model = NLSS(A, B, C, D, E, F)
true_model.add_nl(nlx=nlx, nly=nly)
data2 = simulate(true_model)
res2 = identify(data2, nlx, nly, nmax=nmax, info=info)

figs = plot(res1, data1, p=1)
figs = plot(res2, data2, p=1)
figs = plot_bla(res1, data1, p=1)


"""Output-based nonlinearity"""
tahn1 = Tanhdryfriction(eps=0.1, w=[0,1])
nlx = [tahn1]
F = np.array([])
nly = None

# We get good identification using BLA
E = 1e-1*Efull[:,:len(nlx)]
true_model = NLSS(A, B, C, D, E, F)
true_model.add_nl(nlx=nlx, nly=nly)
data3 = simulate(true_model,npp=2048, Ntr=2)
res3 = identify(data3, nlx, nly, nmax=nmax, info=info)
figs = plot(res3, data3, p=1)
figs = plot_bla(res3, data3, p=1)

# changing the coefficients by a factor of 10, bla is a poor starting guess
# now enter FNSI. FNSI can only use 1 realization
E = 1e0*Efull[:,:len(nlx)]
true_model = NLSS(A, B, C, D, E, F)
true_model.add_nl(nlx=nlx, nly=nly)
data4 = simulate(true_model,npp=2048, Ntr=2)
res4 = identify(data4, nlx, nly, fnsi=True, nmax=nmax, info=info)
figs = plot(res4, data4, p=1)
figs = plot_time(res4, data4, p=1)


"""Can tanh identify sign-nl?"""
class Signfric(Nonlinear_Element):
    """Friction model for simulating sign(ẏ)"""
    def __init__(self, w, **kwargs):
        self.w = np.atleast_1d(w)
        self.n_ny = 1
        super().__init__(**kwargs)

    def set_active(self,n,m,p,q):
        # all are active
        self._active = np.s_[0:q*self.n_nl]
    
    def fnl(self, x,y,u):
        y = np.atleast_2d(y)
        # displacement of dofs attached to nl
        ynl =np.inner(self.w, y)  # (n_nx, ns)
        f = np.sign(ynl)
        return f

    def dfdx(self,x,y,u):
        raise ValueError('This is only for simulating')

    def dfdy(self,x,y,u):
        raise ValueError('This is only for simulating')

# set regularization parameter to something low.
tahn1 = Tanhdryfriction(eps=0.000001, w=[0,1])
nlx = [tahn1]
fric1 = [Signfric(w=[0,1])]
E = 1e-2*Efull[:,:len(fric1)]
true_model = NLSS(A, B, C, D, E, F)
true_model.add_nl(nlx=fric1, nly=nly)
data5 = simulate(true_model,npp=4096, Ntr=5)
res5 = identify(data5, nlx, nly, fnsi=True, nmax=nmax, info=info)
figs = plot_time(res5, data5, p=1)
figs = plot(res5, data5, p=1)
figs = plot_bla(res5, data5, p=1)

# subspace plots
#figs['subspace_optim'] = linmodel.plot_info()
#figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/tutorial_{k}{i}.pdf")

plt.show()
