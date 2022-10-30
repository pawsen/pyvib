#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyvib.common import db, dsample
from pyvib.forcing import multisine, multisine_time, sine, sinesweep
from pyvib.modal import mkc2ss
from pyvib.newmark import Newmark
from pyvib.nlss import NLSS, nlsim2
from pyvib.nonlinear_elements import NLS, Polynomial, Tanhdryfriction
from pyvib.nonlinear_elements_newmark import NLS as nmNLS
from pyvib.nonlinear_elements_newmark import Polynomial as nmPolynomial
from pyvib.nonlinear_elements_newmark import Tanhdryfriction as nmTanhdryfriction
from scipy import signal
from scipy.io import loadmat

"""This script simulates a cantilever beam with attached slider

"""

print(
    "Compare simulation results for: discrete time(d), continous time(c) and newmark(nm) integration"
)


np.seterr(all="raise")

nltype = "pol"
ftype = "sweep"
ftype = "multisine"
savedata = True


if nltype == "pol":
    """Each mass is connected with a polynomial nonlinear spring to ground

    #              fdof
    #            +-->
    #
    #              x1                          x2                        #
    #            +-->                        +-->                        #
    #            |                           |                           #
    #  d1 __     +--------------+     __     +--------------+     __     #
    #-----__|----|              |-----__|----|              |-----__|----#
    #  k1        |              |  k3        |              |  k2        #
    #__/\  /\  __|      M1      |__/\  /\  __|       M2     |__/\  /\  __#
    #    \/  \/  |              |    \/  \/  |              |    \/  \/  #
    # mu1 ^      |              |            |              | mu2 ^      #
    #__/\/ /\  __|              |            |              |__/\/ /\  __#
    #   /\/  \/  +--------------+            +--------------+   /\/  \/  #
    #
    # Mode & Frequency (rad/s) & Damping ratio (%)
    # 1    & 1.00              &       5.00
    # 2    & 3.32              &       1.51

    """
    m1 = 1  # kg
    m2 = 1
    k1 = 1  # N/m
    k2 = 5
    k3 = 1
    c1 = 0.1  # N/ms
    c2 = 0.1

    mu1 = 1  # N/m^3
    mu2 = 1  # N/m^3
    exponent = 3

    w1 = [1, 0]
    w2 = [0, 1]
    nldof = np.array([0, 1])
    nnl = len(nldof)

    f1 = 1e-4 / 2 / np.pi
    f2 = 5 / 2 / np.pi
    fs = 20
    npp = 2**12

    P = 5  # periods
    Avec = [1]

    # specific for sweep
    vsweep = 0.01
    inctype = "lin"

    # specific for multisine
    R = 3  # realizations
    upsamp = 5

    print(
        f"Simulating 2dof with polynomial nonlinearity (exponents {exponent}) and {ftype} forcing"
    )

else:
    """The slider is attached to the end; In order to know the slider velocity,
    needed for output-based identification, the slider is modeled as a small
    extra mass attached to the tip with a spring. The nonlinear damping is then
    found from the extra mass' velocity using a regulized tanh function, ie

    fnl = μ*tanh(ẏ/ε)

    To determine the right multisine amplitude, we make a with scan with increasing
    amplitudes for one period and one realization. By looking at the first modes
    resonance peak in the FRF, we can roughly correlate the amplitude to stick or
    slip condition. We know the natural frequencies for each extreme from linear
    modal analysis, ie. either fully stuck or fully sliding.

    ωₙ = 19.59, 122.17, 143.11  # free
    ωₙ = 21.44, 123.34, 344.78  # stuck

    We need at least 2**13(8192) points per period for good identification of the
    linear system. Even if the system is sampled with more points
    """
    print("error - I can only simulate polynomial nonlin")
    print("please extend this script to simulate tanh")
    sys.exit(0)

# definition of system
fdof = 0
M = np.array([[m1, 0], [0, m2]])
C = np.array([[c1, 0], [0, c2]])
K = np.array([[k1 + k2, -k2], [-k2, k2 + k3]])
M, C, K = np.atleast_2d(M, C, K)
ndof = M.shape[0]

ns = npp * R * P
t = np.arange(ns) / fs
fsint = upsamp * fs
nppint = upsamp * npp
# add extra period which will be removed due to edge effects
Pfilter = 1
if upsamp > 1:
    P = Pfilter + P
nsint = nppint * P * R
dt = 1 / fsint
Ntr = 1

# newmark nonlinearity
nmpol1 = nmPolynomial(w=w1, exp=exponent, k=mu1)
nmpol2 = nmPolynomial(w=w2, exp=exponent, k=mu2)
nmnls = [nmpol1, nmpol2]
newmark = Newmark(M, C, K, nmnls)

# state space
pol1 = Polynomial(exponent=exponent, w=w1)
pol2 = Polynomial(exponent=exponent, w=w2)
nlx = [pol1, pol2]
nly = None

# cont time
a, b, c, d = mkc2ss(M, K, C)
fact = 1
# include velocity in output
if len(w1) == 4:
    c = np.vstack((c, np.hstack((np.zeros_like(M), np.eye(ndof)))))
    d = np.vstack((d, np.zeros_like(M)))
    fact = 2
csys = signal.StateSpace(a, b, c, d)
Ec = np.zeros((2 * ndof, nnl))
Fc = np.zeros((fact * ndof, 0))
Ec[ndof + nldof[0], 0] = -mu1
Ec[ndof + nldof[1], 1] = -mu2

cmodel = NLSS(csys.A, csys.B, csys.C, csys.D, Ec, Fc)
cmodel.add_nl(nlx=nlx, nly=nly)


nhar = 200
f0 = (f2 - f1) / nhar
t2 = P / f0
tc = np.linspace(0, t2, nppint * P, endpoint=False)
# upsampled continious time sampling frequency
fsc = f0 * nppint
freqc = np.arange(nppint) / nppint * fsc

print(f"fs/2: {fs/2:.4g} or fsc/2: {fsc/2:.4g}")
if fsc / 2 <= f2 or fs / 2 <= f2:
    raise ValueError(f"Error, increase fs/2: {fs/2:.4g} or fsc/2: {fsc/2:.4g}")

# convert to discrete time
dsys = csys.to_discrete(dt=dt, method="foh")  # tustin
Ed = np.zeros((2 * ndof, nnl))
Fd = np.zeros((fact * ndof, 0))
# euler discretization
Ed[ndof + nldof[0], 0] = -mu1 * dt
Ed[ndof + nldof[1], 1] = -mu2 * dt

dmodel = NLSS(dsys.A, dsys.B, dsys.C, dsys.D, Ed, Fd, dt=dsys.dt)
dmodel.add_nl(nlx=nlx, nly=nly)

# multisine in time domain
def fex_cont(A, u, t):
    t = np.atleast_1d(t)
    fex = np.zeros((len(t), ndof))
    fex[:, fdof] = A * u(t)
    return fex


def simulate_cont(sys, A, t, fexc=None):
    nt = len(t)
    y = np.empty((R, nt, sys.outputs))
    x = np.empty((R, nt, len(sys.A)))
    u = np.empty((R, nt))
    lines = []
    for r in range(R):
        if ftype == "multisine":
            np.random.seed(r)
            ufunc, lines = multisine_time(f1, f2, N=nhar)
            fexc = partial(fex_cont, A, ufunc)
            u[r] = ufunc(t)

        _, yr, xr = nlsim2(sys, fexc, t=tc)
        y[r] = yr
        x[r] = xr

    return y.reshape((R * nt, -1)), x.reshape((R * nt, -1)), u, lines


if ftype == "multisine":
    # multisine in freq domain
    np.random.seed(10)
    ud, linesd, freqd, t = multisine(f1=f1, f2=f2, N=nppint, fs=fsint, R=R, P=P)
elif ftype == "sweep":
    # freqd is the instantanous forcing frequency here
    ud, tc, freqd = sinesweep(
        amp=1, fs=fsint, f1=f1, f2=f2, vsweep=vsweep, nrep=P, inctype=inctype
    )
    # we dont use the first value, as it is not repeated when the force is
    # generated. This is taken care of in multisine.
    nsper = (len(ud) - 1) // P
elif ftype == "sine":
    ud, tc = sine(A=1, f=f1, fs=fsint, ns=ns)
    # XXX: maybe correct?
    freqd = np.arange(len(tc)) / fsint

else:
    print(f"Wrong type of forcing {ftype}")
    sys.exit(0)


fext = np.zeros((len(ud.ravel()), ndof))

print(f"simulating with ns: {nsint}, R: {R}, P: {P}, upsamp: {upsamp}")
for A in Avec:

    print(f"amplitude {A}")

    # Transient: Add periods before the start of each realization. To generate
    # steady state data.
    T1 = np.r_[npp * Ntr, np.r_[0 : (R - 1) * P * nppint + 1 : P * nppint]]
    # T1 = None
    fext[:, fdof] = A * ud.ravel()
    try:
        print(f"starting discrete time integration.")
        _, yd, xd = dmodel.simulate(fext, T1=T1)
    except Exception as e:
        yd, xd = np.zeros((nppint * R * P, len(w1))), np.zeros(
            (nppint * R * P, 2 * ndof)
        )
        print(f"Discrete simulation failed with error {e}")
    try:
        print("starting continous time integration")
        yc, xc, uc, linesc = simulate_cont(cmodel, A, tc, fext)
    except Exception as e:
        yc, xc = np.zeros((nppint * R * P, len(w1))), np.zeros(
            (nppint * R * P, 2 * ndof)
        )
        uc, linesc = np.zeros((nppint * R * P, ndof)), np.zeros(nhar)
        print(f"Continous simulation failed with error {e}")
    try:
        print("starting newmark integration")
        ynm, ydnm, yddnm = newmark.integrate(
            fext, dt, x0=None, v0=None, sensitivity=False
        )
    except Exception as e:
        ynm = np.zeros((nppint * R * P, len(w1)))
        print(f"Newmark failed with error {e}. For A: {A}")

    print(f"done integrating for A {A}")

    # plot frf for forcing and tanh node
    Yd = np.fft.fft(yd[-nppint:, nldof], axis=0)
    Yc = np.fft.fft(yc[-nppint:, nldof], axis=0)
    Ynm = np.fft.fft(ynm[-nppint:, nldof], axis=0)
    nfd = Yd.shape[0] // 2

    plt.figure()
    plt.plot(freqd[:nfd], db(np.abs(Yd[:nfd])))
    plt.plot(freqd[:nfd], db(np.abs(Ynm[:nfd])))
    plt.plot(freqc[:nfd], db(np.abs(Yc[:nfd])))
    plt.xlim([0, f2])
    # plt.ylim(bottom=-150)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.legend(
        [
            "d: Force dof",
            "d: nl dof",
            "nm: Force dof",
            "nm: nl dof",
            "c: Force dof",
            "c: nl dof",
        ]
    )
    plt.title(
        f"Comparison of simulation results. Remember d: discrete, c: continious, nm: newmark.\nA: {A}"
    )
    plt.minorticks_on()
    plt.grid(which="both")
    # plt.savefig(f'fig/dc_b{benchmark}_A{A}_eps{epsf}_fft_comp_n{fdof}.png')

    # plot displacement (for multisine: only first realization)
    for y, desc in zip([yd, yc, ynm], ("discrete", "continuous", "newmark")):
        plt.figure()
        plt.clf()
        plt.plot(t, y[: len(t), 0], "-k", label=r"$y_1$")
        plt.plot(t, y[: len(t), 1], "-r", label=r"$y_2$")
        plt.xlabel("Time (t)")
        plt.ylabel("Displacement (m)")
        plt.title(f"{desc}. Force type: {ftype}, periods:{P:d}")
        plt.legend()

if savedata:
    try:
        # We need to reshape into (npp,m,R,P)
        if ftype == "multisine":
            if len(w1) != 4:
                yd = np.hstack((yd, yd))
                yc = np.hstack((yc, yc))
            ys = [
                ynm,
                ydnm,
                yddnm,
                yd[:, :ndof],
                yd[:, ndof:],
                yc[:, :ndof],
                yc[:, ndof:],
            ]
            ys = [y.reshape(R, P, nppint, ndof).transpose(2, 3, 0, 1) for y in ys]

            xs = [xd, xc]
            xs = [x.reshape(R, P, nppint, 2 * ndof).transpose(2, 3, 0, 1) for x in xs]

            us = [A * ud, uc]
            us = [u.reshape(R, P, nppint, 1).transpose(2, 3, 0, 1) for u in us]

            # if upsamp:  # > 1:
            #    ys = [dsample(y, upsamp, zero_phase=True) for y in ys]
            #    xs = [dsample(y, upsamp, zero_phase=True) for y in xs]
            #    us = [u[::upsamp, :, :, 1:] for u in us]
        elif ftype == "pol":
            pass

        filename = f"data/{nltype}_{ftype}_A{A}_upsamp{upsamp}_fs{fs}.npz"
        # ennsure the folder exists
        Path(filename).parent.mkdir(exist_ok=True)

        np.savez(
            filename,
            ynm=ys[0],
            ydotnm=ys[1],
            yddotnm=ys[2],
            yd=ys[3],
            ydotd=ys[4],
            xd=xs[0],
            ud=us[0],
            linesd=linesd,
            yc=ys[5],
            ydotc=ys[6],
            xc=xs[1],
            uc=us[1],
            linesc=linesc,
            fs=fsint,
            A=A,
            fsc=fsc,
        )
        print(f"data saved as {filename}")
    except:
        print(f"data not saved due to an error: {{XXX}}")

# plt.figure()
# plt.plot(t, x, '-k', label=r'$x_1$')
##plt.plot(t, x, '-r', label=r'$x_2$')
# plt.xlabel('Time (t)')
# plt.ylabel('Displacement (m)')
# plt.title('Force type: {}, periods:{:d}')
# plt.legend()
#
# plt.figure()
# plt.plot(np.abs(np.fft.fft(x[6*1024:7*1024,0])))
#
#
# x = ufunc(tc)
# X = np.fft.fft(x)
# nfd = X.shape[0]//2
# plt.figure()
# plt.plot(freq[:nfd], db(np.abs(X[:nfd])))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude (dB)')


plt.show()
