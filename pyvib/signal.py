#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
import numpy as np
from numpy.fft import fft
from scipy.signal import decimate

from .common import db, prime_factor
from .filter import differentiate, integrate
from .frf import bla_periodic, covariance
from .helper.plotting import periodicity

class Signal():
    def __init__(self, u, y, yd=None, fs=1):
        # in case there is only one realization, ie. (npp,m,P)
        if len(u.shape) == 3:
            u = u[:,:,None]
        if len(y.shape) == 3:
            y = y[:,:,None]
        if yd is not None and len(yd.shape) == 3:
            yd = yd[:,:,None]
        self.u = u
        self.y = y
        self._yd = yd
        self._ydm = None
        self.fs = fs
        self.npp, self.m, self.R, self.P = u.shape
        self.npp, self.p, self.R, self.P = y.shape
        self._lines = None
        self._covY = None

    @property
    def lines(self):
        return self._lines

    @lines.setter
    def lines(self, lines):
        self._lines = lines
        self.F = len(lines)
        self.norm_freq = lines/self.npp  # Excited frequencies (normalized)

    def bla(self):
        """Get best linear approximation"""
        # TODO bla expects  m, R, P, F = U.shape
        self.U = fft(self.u, axis=0)[self.lines].transpose((1,2,3,0))
        self.Y = fft(self.y, axis=0)[self.lines].transpose((1,2,3,0))
        self.G, self.covG, self.covGn = bla_periodic(self.U, self.Y)
        self.G = self.G.transpose((2,0,1))
        if self.covG is not None:
            self.covG = self.covG.transpose((2,0,1))
        if self.covGn is not None:
            self.covGn = self.covGn.transpose((2,0,1))
        return self.G, self.covG, self.covGn

    @property
    def covY(self):
        if self._covY is None:
            self._covY = covariance(self.y)
        return self._covY

    def average(self, u=None, y=None):
        """Average over periods and flatten over realizations"""

        saveu = False
        savey = False
        if u is None:
            u = self.u
            saveu = True
        if y is None:
            y = self.y
            savey = True
        um = u.mean(axis=-1)  # (npp,m,R)
        ym = y.mean(axis=-1)
        um = um.swapaxes(1,2).reshape(-1,self.m, order='F')  # (npp*R,m)
        ym = ym.swapaxes(1,2).reshape(-1,self.p, order='F')  # (npp*R,p)

        if saveu:
            self.um = um
            # number of samples after average over periods
            self.mns = um.shape[0]  # mns = npp*R
        if savey:
            self.ym = ym

        return um, ym

    @property
    def ydm(self):
        if self._yd is None:
            # TODO do some numerical differentiation
            pass
        if self._ydm is None:
            ydm = self._yd.mean(axis=-1)
            self._ydm = ydm.swapaxes(1,2).reshape(-1,self.p, order='F')
        return self._ydm  # (npp*R,m)

    def periodicity(self, dof=0, R=0, P=None, n=1, fig=None, ax=None, **kwargs):
        """Shows the periodicity for the signal"""

        return periodicity(y=self.y, fs=self.fs, dof=dof, R=R, P=P, n=n,
                           fig=fig, ax=ax, **kwargs)

def downsample(y, u, n, nsper=None, keep=False):
    """Filter and downsample signals

    The displacement is decimated(low-pass filtered and downsampled) where
    forcing is only downsampled by the sampling factor n, ie. every n'th
    sample is kept.

    Parameters
    ----------
    n : int
        downsample rate, ie. keep every n'th sample
    keep : bool, optional
        Remove the last period to eliminate the edge effects due to the
        low-pass filter.

    """
    # axis to operate along
    axis = 0

    # filter and downsample
    # prime factor decomposition.
    for k in prime_factor(n):
        y = decimate(y, q=k, ftype='fir', axis=axis)

    # index for downsampling u
    sl = [slice(None)] * u.ndim
    sl[axis] = slice(None, None, n)
    u = u[sl]

    # Removal of the last simulated period to eliminate the edge effects
    # due to the low-pass filter.
    if not keep:
        y = y[...,:-1]
        u = u[...,:-1]

    return u, y

def filt(self, lowcut, highcut, order=3):
    from scipy import signal
    fn = 0.5 * self.fs
    if highcut > fn:
        raise ValueError('Highcut frequency is higher than nyquist\
        frequency of the signal', highcut, fn)
    elif lowcut <= 0:
        raise ValueError('Lowcut frequency is 0 or lower', lowcut, fn)

    b, a = signal.butter(order, highcut, btype='lowpass')
    return signal.filtfilt(b, a, self.y)



class Signal2(object):
    """ Holds common properties for a signal
    """
    def __init__(self, u, fs, y=None, yd=None, ydd=None):
        """
        Parameters
        ----------
        y : ndarray(ns, ndof)
            accelerations
        fs : int
            Sampling frequency

        """
        # cast to 2d. Format is now y[ndofs,ns]. For 1d cases ndof=0
        self.y = _set_signal(y)
        self.yd = _set_signal(yd)
        self.ydd = _set_signal(ydd)

        self.isset_y = False
        self.isset_yd = False
        self.isset_ydd = False
        # ns: total sample points
        if y is not None:
            self.ndof, self.ns = self.y.shape
            self.isset_y = True
        elif yd is not None:
            self.ndof, self.ns = self.yd.shape
            self.isset_yd = True
        elif ydd is not None:
            self.ndof, self.ns = self.ydd.shape
            self.isset_ydd = True

        self.u = _set_signal(u)
        self.fs = fs

        self.y_per = None
        self.u_per = None
        self.iscut = False
        self.isnumeric = False

    def cut(self, nsper, per, offset=0):
        """Extract periodic signal from original signal

        Parameters
        ----------
        nsper : int
            Number of samples per period
        per : list
            List of periods to use. 0-based. Ie. [0,1,2] etc
        """
        per = np.atleast_1d(per)
        # number of periods. Only used for this check
        ns = self.ns
        _nper = int(np.floor(ns / nsper))
        if any(p > _nper - 1 for p in per):
            raise ValueError('Period too high. Only {} periods in data.'.
                             format(_nper),per)

        self.iscut = True
        self.nper = len(per)
        self.nsper = int(nsper)
        # number of sample for cut'ed signal
        ns = self.nper * self.nsper

        # extract periodic signal
        if self.isset_y:
            self.y_per = _cut(self.y, per, self.nsper,offset)
        if self.isset_yd:
            self.yd_per = _cut(self.yd, per, self.nsper,offset)
        if self.isset_ydd:
            self.ydd_per = _cut(self.ydd, per, self.nsper,offset)

        self.u_per = _cut(self.u, per, self.nsper, offset)


    def periodicity(self, nsper=None, dof=0, offset=0, fig=None, ax=None,
                    **kwargs):
        """Shows the periodicity for the signal


        Parameters:
        ----------
        nsper : int
            Number of points per period
        dof : int
            DOF where periodicity is plotted for
        offset : int
            Use offset as first index
        """
        if nsper is None:
            nsper = self.nsper

        fs = self.fs

        y = self.y[dof,offset:]
        ns = len(y)
        ndof = self.ndof
        # number of periods
        nper = int(np.floor(ns / nsper))
        # in case the signal contains more than an integer number of periods
        ns = nper*nsper
        t = np.arange(ns)/fs

        # first index of last period
        ilast = ns - nsper

        # reference period. The last measured period
        yref = y[ilast:ilast+nsper]
        yscale = np.amax(y) - np.amin(y)
        # holds the similarity of the signal, compared to reference period
        va = np.empty(nsper*(nper-1))

        nnvec = np.arange(-nper,nper+1, dtype='int')
        va_per = np.empty(nnvec.shape)
        for i, n in enumerate(nnvec):
            # index of the current period. Moving from 1 to nn-1 because of
            # if/break statement
            ist = ilast + n * nsper
            if ist < 0:
                continue
            elif ist > ns - nsper - 1:
                break

            idx = np.arange(ist,ist+nsper)
            # difference between signals in dB
            va[idx] = db(y[idx] - yref)
            va_per[i] = np.amax(va[idx])

        signal = db(y[:ns])
        if fig is None:
            fig, ax = plt.subplots()
            ax.clear()

        ax.set_title('Periodicity of signal for DOF {}'.format(dof))
        ax.plot(t,signal,'--', label='signal', **kwargs)  # , rasterized=True)
        ax.plot(t[:ilast],va, label='periodicity', **kwargs)
        for i in range(1,nper):
            x = t[nsper * i]
            ax.axvline(x, color='k', linestyle='--')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel(r'$\varepsilon$ dB')
        ax.legend()

        return fig, ax

    def get_displ(self, lowcut, highcut, isnumeric=False):
        """Integrate signals to get velocity and displacement"""

        self.isnumeric = isnumeric
        ydd = self.ydd
        ndof = self.ndof
        fs = self.fs

        y = np.empty(ydd.shape)
        yd = np.empty(ydd.shape)
        for i in range(ndof):
            y[i,:], yd[i,:] = integrate(ydd[i,:], fs, lowcut, highcut,
                                        isnumeric=isnumeric)

        self.isset_y = True
        self.isset_yd = True
        self.y = y
        self.yd = yd

    def get_accel(self, isnumeric=False):
        """ Differentiate signals to get velocity and accelerations
        """
        self.isnumeric = isnumeric
        y = self.y
        ndof = self.ndof
        fs = self.fs

        yd = np.empty(y.shape)
        ydd = np.empty(y.shape)
        for i in range(ndof):
            yd[i,:], ydd[i,:] = differentiate(y[i,:], fs, isnumeric=isnumeric)

        self.yd = yd
        self.ydd = ydd

    def set_signal(self, y=None, yd=None, ydd=None):
        self.y = _set_signal(y)
        self.yd = _set_signal(yd)
        self.ydd = _set_signal(ydd)
        if y is not None:
            self.ndof, self.ns = self.y.shape
            self.isset_y = True
        elif yd is not None:
            self.ndof, self.ns = self.yd.shape
            self.isset_yd = True
        elif ydd is not None:
            self.ndof, self.ns = self.ydd.shape
            self.isset_ydd = True
    # def wt(self, f1, f2, nf=50, f00=10, dof=0, pad=0):
    #     fs = self.fs
    #     x = self.y[dof]
    #     finst, wtinst, time, freq, y = morletWT(x, fs, f1, f2, nf, f00, pad)

    #     # poor mans object
    #     WT = namedtuple('WT', 'f1 f2 nf f00  dof pad finst wtinst time freq y')
    #     self.wt = WT(f1, f2, nf, f00, dof, pad,finst, wtinst, time, freq, y)

def _set_signal(y):
    if y is not None:
        y = np.atleast_2d(y)
        return y
    return None

def _cut(x,per,nsper,offset=0):
    """Extract periodic signal from original signal"""

    nper = len(per)
    ndof = x.shape[0]
    x_per = np.empty((ndof, nper*nsper))

    for i, p in enumerate(per):
        x_per[:,i*nsper: (i+1)*nsper] = x[:,offset + p*nsper:
                                          offset+(p+1)*nsper]
    return x_per



# def derivative(Y):
#     # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.diff.html
#     N = npp
#     if N % 2 == 0:
#         # NOTE k is sequence of ints
#         k = np.r_[np.arange(0, N//2), [0], np.arange(-N//2+1, 0)]
#     else:
#         k = np.r_[np.arange(0, (N-1)//2), [0], np.arange(-(N-1)//2, 0)]

#     freq = self.flines / self.npp
#     k *= 2 * np.pi / freq
#     yd = np.real(np.fft.ifft(1j*k*Y*np.sqrt(npp),axis=0))

#     return yd
