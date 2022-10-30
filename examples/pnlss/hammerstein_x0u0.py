#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pyvib.pnlss import PNLSS
from pyvib.signal import Signal

print("""Estimating Hammerstein system with and without estimated non-zero initial condition.""")

np.random.seed(10)
N = int(2e3)  # Number of samples
NTrans = 100  # Number of samples after zero initial conditions
u = np.random.randn(NTrans + N)


def f_NL(x):
    return x + 0.2 * x**2 + 0.1 * x**3


b, a = signal.cheby1(2, 5, 2 * 0.3, "low", analog=False)
x = f_NL(u)  # Intermediate signal
y = signal.lfilter(b, a, x)  # output
# remove first NTrans samples to obtain non-zero initial conditions
mask = np.ones(len(u), dtype=bool)
mask[np.r_[:NTrans]] = False
u = u[mask]
x = x[mask]
y = y[mask]
scale = np.linalg.lstsq(u[:, None], x)[0].item()
# Initial linear model = scale factor times underlying dynamics
sys = signal.tf2ss(scale * b, a)
T1 = 0  # No periodic transient handling
# XXX: transient handling doesnt work, throws an error if T2 != 0
T2 = 0  # No transient samples to discard
sig = Signal(u[:, None, None, None], y[:, None, None, None])
sig.average()
model = PNLSS(sys, sig)

# Hammerstein system only has nonlinear terms in the input.
# Quadratic and cubic terms in state- and output equation
model.nlterms("x", [2, 3], "inputsonly")
model.nlterms("y", [2, 3], "inputsonly")
model.transient(T1=T1, T2=T2)

print("Optimizing without estimated initial conditions")
model.optimize(weight=False, nmax=50)
yopt = model.simulate(u)[1].squeeze()

# Estimate initial conditions. Start from nonlinear model
model_x0u0 = deepcopy(model)
print("Optimizing with estimated initial conditions")
model_x0u0.optimize(weight=False, nmax=50)
yopt_x0u0 = model_x0u0.simulate(u)[1].squeeze()


def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))


print(f"RMS error without init cond estimated {rmse(y, yopt)}")
print(f"RMS error with init cond estimated {rmse(y, yopt_x0u0)}")

err_noinit = y - yopt
err_init = y - yopt_x0u0
t = np.arange(N)

plt.figure()
plt.plot(t, err_noinit - err_init)
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend(
    ["Difference in error between PNLSS with and without estimated inittial conditions"]
)

plt.figure()
plt.title("Output and errors")
plt.plot(t, y)
plt.plot(t, err_noinit, ".", lw=3)
plt.plot(t, err_init, "--k", lw=3)
plt.xlabel("Time")
plt.ylabel("Output")
plt.legend(["Output", "Error PNLSS", "Error PNLSS (initial conditions estimated)"])

plt.show()
