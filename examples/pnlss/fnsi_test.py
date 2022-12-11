#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm

from scipy.io import loadmat

from pyvib.common import db
from pyvib.forcing import multisine
from pyvib.frf import covariance
from pyvib.nlss import NLSS
from pyvib.nonlinear_elements import (NLS, Nonlinear_Element,
                                      Tanhdryfriction, Unilatteralspring)
from pyvib.signal import Signal
from pyvib.subspace import Subspace
from pyvib.fnsi import FNSI

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
savefig = False
add_noise = False
weight = False
p = 2

## Generate data from true model ##
# Construct model to estimate
A = np.array([[0.73915535, -0.62433133],[0.6247377, 0.7364469]])
B = np.array([[0.79287245], [-0.34515159]])
C = np.array([[0.71165154, 0.34917771]])
D = np.array([[0.04498052]])
if p == 2:
    C = np.vstack((C,C))
    D = np.vstack((D,0.1563532))

Ffull = np.array([[-0.00867042, -0.00636662]])
Efull = np.array([[1.88130305e-01, -2.70291900e-01],
                  [-5.35196110e-01,-3.66250013e-01]])


if p == 1:
    Wy = [1]
    Wt = [1]
elif p ==2:
    Wt = [0,1]


F = np.array([])
nly = None

# for identification
tahn1 = Tanhdryfriction(eps=0.1, w=Wt)
nlx = [tahn1]
E = 1e0*Efull[:,:len(nlx)]

true_model = FNSI(A, B, C, D, E, F)
true_model.add_nl(nlx=nlx, nly=nly)

# excitation signal
RMSu = 0.05   # Root mean square value for the input signal
npp = 2048    # Number of samples
R = 3         # Number of phase realizations (one for validation and one for
              # testing)
P = 3         # Number of periods
kind = 'Odd'  # 'Full','Odd','SpecialOdd', or 'RandomOdd': kind of multisine
m = D.shape[1]         # number of inputs
p = C.shape[0]         # number of outputs
fs = 1        # normalized sampling rate
Ntr = 2
if True:
    # get predictable random numbers. https://dilbert.com/strip/2001-10-25
    np.random.seed(10)
    # shape of u from multisine: (R,P*npp)
    u, lines, freq, t = multisine(N=npp, P=P, R=R, lines=kind, rms=RMSu)
    # if multiple input is required, this will copy u m times

    # Transient: Add one period before the start of each realization. To generate
    # steady state data.
    T1 = np.r_[npp*Ntr, np.r_[0:(R-1)*P*npp+1:P*npp]]
    _, yorig, _ = true_model.simulate(u.ravel(), T1=T1)
    print(norm(yorig))
    u = u.reshape((R,P,npp)).transpose((2,0,1))[:,None]  # (npp,m,R,P)
    y = yorig.reshape((R,P,npp,p),order='C').transpose((2,3,0,1))

    #y = yorig.reshape((R,P,npp)).transpose((2,0,1))[:,None]
    # or in F order:
    # y2 = yorig.reshape((npp,P,R,p),order='F').transpose((0,3,2,1))
    
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
    uest = u[...,0,:][:,:,None]
    yest = y[...,0,:][:,:,None]
    # noise estimate over periods. This sets the performace limit for the estimated
    # model
    covY = covariance(yest)
    Pest = yest.shape[-1]
    
    # create signal object
    sig = Signal(uest,yest,fs=fs)
    sig.lines = lines
    # plot periodicity for one realization to verify data is steady state
    #sig.periodicity()
    # Calculate BLA, total- and noise distortion. Used for subspace identification
    sig.bla()
    # average signal over periods. Used for training of PNLSS model
    um, ym = sig.average()

# model orders and Subspace dimensioning parameter
nvec = [2,3]
maxr = 5


if 'linmodel' not in locals() or True:
    Rest = yest.shape[2]
    T1 = np.r_[npp*Ntr, np.r_[0:(Rest-1)*npp+1:npp]]
    
    linmodel = FNSI()
    linmodel.set_signal(sig)
    linmodel.estimate(n=2, r=5, weight=weight)

    linmodel.transient(T1)
    # linmodel._cost_normalize = 1
    linmodel.optimize(weight=weight)
    
    fnsi1 = FNSI()
    fnsi1.set_signal(sig)
    fnsi1.add_nl(nlx=nlx)
    fnsi1.estimate(n=2, r=5, weight=weight)
    fnsi1.transient(T1)
    #fnsi1.optimize(lamb=100, weight=weight, nmax=25)
    
    print(f"Best subspace model, n, r: {linmodel.n}, {linmodel.r}")
    # linmodel_orig = linmodel

if False:  # dont scan subspace
    linmodel = Subspace(sig)
    # get best model on validation data
    models, infodict = linmodel.scan(nvec, maxr, weight=weight)
    l_errvec = linmodel.extract_model(yval, uval)
    # or estimate the subspace model directly
    linmodel.estimate(2, 5, weight=weight)  # best model, when noise weighting is used
    linmodel.optimize(weight=weight)
    print(f"Best subspace model, n, r: {linmodel.n}, {linmodel.r}")
    
# linmodel = deepcopy(linmodel_orig)
# estimate PNLSS
# transient: Add one period before the start of each realization. Note that
# this is for the signal averaged over periods
Rest = yest.shape[2]
T1 = np.r_[npp*Ntr, np.r_[0:(Rest-1)*npp+1:npp]]


nlx2 = NLS([tahn1])
nly2 = None


model = NLSS(linmodel)
model._cost_normalize = 1
model.add_nl(nlx=nlx2, nly=nly2)
model.set_signal(sig)
model.transient(T1)
model.optimize(lamb=100, weight=weight, nmax=25)

#raise SystemExit(0)

# get best model on validation data. Change Transient settings, as there is
# only one realization
nl_errvec = model.extract_model(yval, uval, T1=npp*Ntr)

models = [linmodel, model, fnsi1]
descrip = [type(mod).__name__ for mod in models]
descrip = tuple(descrip)  # convert to tuple for legend concatenation in figs
# simulation error
val = np.empty((*yval.shape, len(models)))
est = np.empty((*ym.shape, len(models)))
test = np.empty((*ytest.shape, len(models)))
for i, model in enumerate(models):
    test[...,i] = model.simulate(utest, T1=npp*Ntr)[1]
    val[...,i] = model.simulate(uval, T1=npp*Ntr)[1]
    est[...,i] = model.simulate(um, T1=T1)[1]

# convenience inline functions
stack = lambda ydata, ymodel: \
    np.concatenate((ydata[...,None], (ydata[...,None] - ymodel)),axis=2)
rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
est_err = stack(ym, est)  # (npp*R,p,nmodels)
val_err = stack(yval, val)
test_err = stack(ytest, test)
noise = np.abs(np.sqrt(Pest*covY.squeeze()))
print(f"err for models {descrip}")
print(f'rms error noise: {rms(noise)}\tdb: {db(rms(noise))} ')
print(f'rms error est:  \n{rms(est_err)}   \ndb: \n{db(rms(est_err))}')
print(f'rms error val:  \n{rms(val_err)}   \ndb: \n{db(rms(val_err))}')
print(f'rms error test: \n{rms(test_err)}  \ndb: \n{db(rms(test_err))}')


## Plots ##
# store figure handle for saving the figures later
figs = {}

# linear and nonlinear model error; plot for each output
for pp in range(p):
    plt.figure()
    plt.plot(est_err[:,pp])
    plt.xlabel('Time index')
    plt.ylabel('Output (errors)')
    plt.legend(('Output',) + descrip)
    plt.title(f'Estimation results p:{pp}')
    figs['estimation_error'] = (plt.gcf(), plt.gca())

    # result on validation data
    N = len(yval)
    freq = np.arange(N)/N*fs
    plottime = val_err
    plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
    nfd = plotfreq.shape[0]
    plt.figure()
    plt.plot(freq[lines], db(plotfreq[lines,pp]), '.')
    plt.plot(freq[lines], db(np.sqrt(Pest*covY[lines,pp,pp].squeeze() / N)), '.')
    plt.xlabel('Frequency')
    plt.ylabel('Output (errors) (dB)')
    plt.legend(('Output',) + descrip + ('Noise',))
    plt.title(f'Validation results p:{pp}')
    figs['val_data'] = (plt.gcf(), plt.gca())

    # result on test data
    N = len(ytest)
    freq = np.arange(N)/N*fs
    plottime = test_err
    plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
    nfd = plotfreq.shape[0]
    plt.figure()
    plt.plot(freq[:nfd//2], db(plotfreq[:nfd//2,pp]), '.')
    plt.plot(freq[:nfd//2], db(np.sqrt(Pest*covY[:nfd//2,pp,pp].squeeze() / N)), '.')
    plt.xlabel('Frequency')
    plt.ylabel('Output (errors) (dB)')
    plt.legend(('Output',) + descrip + ('Noise',))
    plt.title(f'Test results p:{pp}')
    figs['test_data'] = (plt.gcf(), plt.gca())
    
    try:
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
        bla_noise = db(np.sqrt(np.abs(sig.covGn[:,pp,pp])*R))
        bla_noise[bla_noise < -150] = -150
        bla_tot = db(np.sqrt(np.abs(sig.covG[:,pp,pp])*R))
        bla_tot[bla_tot < -150] = -150
    
        plt.plot(freq[lines], db(np.abs(sig.G[:,pp,0])))
        plt.plot(freq[lines], bla_noise,'s')
        plt.plot(freq[lines], bla_tot,'*')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('magnitude (dB)')
        plt.title(f'Estimated BLA p: {pp}')
        plt.legend(('BLA FRF','Noise Distortion','Total Distortion'))
        plt.gca().set_ylim(bottom=-150) 
        figs['bla'] = (plt.gcf(), plt.gca())
    except:
        pass

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
#figs['subspace_optim'] = linmodel.plot_info()
#figs['subspace_models'] = linmodel.plot_models()

if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/tutorial_{k}{i}.pdf")

plt.show()


"""
Workable parameters
-------------------
RMSu = 0.05
Ntr = 5
nlx = NLS([Tanhdryfriction(eps=0.1, w=[1])])
E = 1e-1*Efull[:,:len(nlx.nls)]

----
RMSu = 0.05
Ntr = 5

gap = 0.25
nlx = NLS([Unilatteralspring(gap=gap, w=[1])])
E = 1e-1*Efull[:,:len(nlx.nls)]
----

RMSu = 0.05
Ntr = 5
E = Efull[:,:2]
nlx = NLS([poly2y, poly1y])  #, poly3])

nlx2 = NLS([poly1y,poly3y,poly2x,poly2y])  #,poly3])
nly2 = None

====
p = 2

if p == 2:
    C = np.vstack((C,C))
    D = np.vstack((D,0.1563532))

E = Efull
F = Ffull
nlx = NLS([Pnlss(degree=[2,3], structure='full')])
nly = NLS([Pnlss(degree=[2,3], structure='full')])
----

elif p ==2:
    Wy = np.array([[1,0],[0,1]])
    exp1 = [2,1]
    exp2 = [2,2]
    exp3 = [3,1]


nly = None
nlx = NLS([poly2y, poly1y])
E = Efull[:,:len(nlx.nls)]

nlx2 = NLS([poly1y,poly3y,poly2x,poly2y])
"""
