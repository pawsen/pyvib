#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fftpack import fft, ifft
from .common import next_pow2, db

class WT():
    def __init__(self, signal):
        self.signal = signal

    def morlet(self, f1, f2, nf=50, f00=10, dof=0, pad=0):
        self.f1 = f1
        self.f2 = f2
        self.nf = nf
        self.f00 = f00
        self.dof = dof
        self.pad = pad

        fs = self.signal.fs
        x = self.signal.y[dof]
        finst, wtinst, time, freq, y = morletWT(x, fs, f1, f2, nf, f00, pad)
        self.finst = finst
        self.wtinst = wtinst
        self.time = time
        self.freq = freq
        self.y = y

    def plot(self, fss=None, sca=1, **kwargs):

         fig, ax = waveletPlot(self.finst, self.wtinst, self.time, self.freq,
                               self.y, fss, sca, **kwargs)
         self.fig = fig
         self.ax = ax
         return fig, ax


def morletWT(x, fs, f1, f2, nf, f00, pad=0):
    """

    Parameters
    ----------
    x: ndarray
        Displacements (or velocities or accelerations) for a single DOF
    fs: float
        Sampling frequency
    nf: int
        Frequency steps
    f00: float in range [2-20]
        Morlet coefficient
    pad: int
        Padding

    Returns
    -------
    finst: ndarray, len(x)

    wtinst: ndarray, len(x)

    time: ndarray, len(x)
        Time for wt, ie. x-axis
    freq: ndarray, len:nf
        Instantaneous frequency, ie. y-axis
    y: ndarray [nf, len(x)]
        FFT Amplitudes. Stored as [Freq, time]. Ie most likely to be used as y.T
    """
    x = np.squeeze(x)

    dt = 1/fs
    df = (f2 - f1) / nf
    freq = np.linspace(f1, f2, nf)

    a = f00 / (f1 + np.outer(np.arange(nf),df))
    na = len(a) - 1

    k = 2**pad
    NX = len(x)
    NX2 = next_pow2(NX)
    N = 2**NX2
    N = k*N

    time = np.arange(N)*dt
    f = np.linspace(0, fs/2, N//2)
    omega = f*2*np.pi

    filt = np.sqrt(2*a @ np.ones((1,N//2))) * \
        np.exp(-0.5*(a @ omega[None,:] - 2*np.pi*f00)**2)
    filt[np.isnan(filt)] = 0

    X = fft(x, N, axis=0)
    X = np.conj(filt) * (np.ones((na+1,1)) @ X[None,:N//2])
    y = np.zeros((na+1,N), dtype=complex)
    for j in range(na+1):
        y[j] = ifft(X[j], N)

    y = y.T
    mod = np.abs(y)

    imax = np.argmax(mod, axis=1)
    wtinst = np.max(mod, axis=1)
    finst = f00 / a[imax].squeeze()
    finst = finst[:NX]
    wtinst = wtinst[:NX]
    y = y[:NX]
    time = time[:NX]

    return finst, wtinst, time, freq, y


def waveletPlot(finst, wtinst, time, freq, y, fss=None, sca=1, **kwargs):

    if sca == 1:
        unit =  ' (Hz)'
    else:
        unit =  ' (rad/s)'

    if fss is None:
        vx = time
        xstr = 'Time (s)'
    else:
        vx = fss*sca
        xstr = 'Sweep frequency' + unit

    # Some textture settings. Used to reduce the textture size, but not needed
    # for now.
    nmax = len(freq) if len(freq) > len(time) else len(time)
    n1 = len(freq) // nmax
    n1 = 1 if n1 < 1 else n1
    n2 = len(vx) // nmax
    n2 = 1 if n2 < 1 else n2

    freq = freq[::n1]*sca
    vx = vx[::n2]
    finst = finst[::n2]
    y = y[::n2,::n1]

    T, F = np.meshgrid(vx, freq)
    va = db(y)
    va[va < - 200] = -200

    # fig = plt.figure(1)
    # plt.clf()
    # ax = fig.add_subplot(111)
    fig, ax = plt.subplots()

    extends = ["neither", "both", "min", "max"]
    cmap = plt.cm.get_cmap("jet")
    cmap.set_under("white")
    cmap.set_over("yellow")

    cs = ax.contourf(T, F, va.T, 10, cmap=cmap, extend=extends[0])
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    ax2 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    # obtain the colormap limits
    vmin,vmax = cs.get_clim()
    # Define a normalised scale
    cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # Plot the colormap in the created axes
    cbar = mpl.colorbar.ColorbarBase(ax2, norm=cNorm, cmap=cmap)
    fig.subplots_adjust(left=0.05,right=0.85)

    cbar.ax.set_ylabel('Amplitude (dB)')
    ax.set_xlabel(xstr)
    ax.set_ylabel('Instantaneous frequency' + unit)

    return fig, ax
