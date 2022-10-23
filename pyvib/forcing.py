#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import ifft
from typing import Optional

"""
Example of a closure-function. See also partial from ...
def force(A, f, ndof, fdof):
    # Closure function. See also functools.partial
    # fext = self.force(dt, tf=T)
    def wrapped_func(dt, t0=0, tf=1):
        ns = round((tf-t0)/dt)
        fs = 1/dt

        u,_ = sine(A, f=f, fs=fs, ns=ns, phi_f=0)
        fext = toMDOF(u, ndof, fdof)
        return fext
    return wrapped_func

"""


def sinesweep(amp:float, fs:float, f1:float, f2:float, vsweep:float, nrep:int=1,
              inctype:str='lin', t0:float=0):
    """Do a linear or logarithmic sine sweep excitation.

    For a reverse sweep, swap f1 and f2 and set a negative sweep rate.

    Parameters
    ----------
    amp : float
        Amplitude in N
    fs : float
        Sampling frequency
    f1 : float
        Starting frequency in Hz
    f2 : float
        Ending frequency in Hz
    vsweep : float
        Sweep rate in Hz/min
    nrep : int
        Number of times the signal is repeated
    inctype : str (optional)
        Type of increment. Linear or logarithmic: lin/log
    t0 : float (optional)
        Staring time, default t0=0

    Notes
    -----
    See scipy.signal.chirp, which does the same
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html
    """
    dt = 1/fs

    if inctype == 'log':
        tend = np.log2(f2 / f1) * (60/vsweep) + t0
    else:
        tend = (f2 - f1) / vsweep * 60 + t0

    # Because we want to enforce the given fs, arange is used instead of
    # linspace. This means that we might not include tend in t (which would be
    # the case with linspace), but for that we get the desired fs.
    ns = int(np.floor((tend-t0)*fs))
    t = np.arange(0, ns+1)/fs
    # t = np.linspace(t0, tend, ns +1)

    # Instantaneous frequency
    if inctype == 'log':
        finst = f1 * 2**(vsweep*((t - t0)/60))
    else:
        finst = f1 + vsweep/60*(t-t0)

    if inctype == 'log':
        psi = (2*np.pi * f1*60/(np.log(2)*vsweep)) * \
            (2**(vsweep*((t-t0)/60)) - 1)
    else:
        psi = 2*np.pi * f1*(t-t0) + 2*np.pi*vsweep/60*(t-t0)**2 / 2

    u = amp * np.sin(psi)
    if nrep > 1:
        # repeat signal: 1 2 3 -> 1 2 3 1 2 3 1 2 3
        u = np.tile(u, nrep)
        # prevent the first number from reoccurring: 1 2 3 -> 1 2 3 2 3 2 3
        idx = np.arange(1, nrep) * (ns+1)
        u = np.delete(u, idx)
        t = np.arange(0, ns*nrep+1) / fs

    return u, t, finst


def multisine_time(f1, f2, N, ms_type='full'):
    """Multisine in time domain

    Be sure to include integer number of periods when creating the time vector

    Parameters
    ----------
    N : int
       Number of harmonics

    Returns
    -------
    fex : callable
        Callable that takes time scalar or array as input

    Examples
    --------
    >>> f1 = 0
    >>> f2 = 150
    >>> N = 100
    >>> fex = multisine_time(f1=f1, f2=f2, N=N)
    >>> P = 1
    >>> f0 = (f2-f1)/N
    >>> t2 = P/f0
    >>> Nt = 1000
    >>> t = np.linspace(0, t2, Nt*P, endpoint=False)
    >>> x = fex(t)
    >>> fs = f0*Nt
    >>> X = np.fft.fft(x)
    >>> nfd = X.shape[0]//2
    >>> freq = np.arange(Nt)/Nt * fs
    >>> plt.figure()
    >>> plt.plot(freq[:nfd], db(np.abs(X[:nfd])))
    >>> plt.xlabel('Frequency (Hz)')
    >>> plt.ylabel('Amplitude (dB)')

    """
    f0 = (f2-f1) / N
    linesMin = np.ceil(f1/f0).astype(int)
    linesMax = np.floor(f2/f0).astype(int)
    lines = np.arange(linesMin, linesMax, dtype=int)

    # remove DC line
    if lines[0] == 0:
        lines = lines[1:]

    phases = 2*np.pi*np.random.rand(len(lines))

    def fex(t):
        try:
            return (np.sum(np.cos(2*np.pi*f0*np.outer(t, lines) + phases),
                           axis=1) / np.sqrt(len(lines)))
        # if t or lines are too long, the outer product result in memory error
        except MemoryError:
            f = 0
            for phase, line in zip(phases, lines):
                f += np.cos(2*np.pi*f0*t*line + phase)
            return f / np.sqrt(len(lines))

    return fex, lines


def multisine(f1: float=0, f2:Optional[float]=None, N:int=1024, fs:Optional[float]=None, R:int=1, P:int=1, lines='full', rms:float=1, ngroup:int=4):
    """Random periodic excitation

    Generates R realizations of a zero-mean random phase multisine with
    specified rms(amplitude). Random phase multisine signal is a periodic
    random signal with a user-controlled amplitude spectrum and a random phase
    spectrum drawn from a uniform distribution. If an integer number of periods
    is measured, the amplitude spectrum is perfectly realized, unlike classical
    Gaussian noise. Another advantage is that the periodic nature can help help
    separate signal from noise.

    The amplitude spectrum is flat between f1 and f2.

    Parameters
    ----------
    f1 : float, optional
        Starting frequency in Hz. Default 0 Hz
    f2 : float, optional
        Ending frequency in Hz. Default 0.9* `nyquist frequency`
    N : int, optional
        Number of points per period. default = 1024
    fs : float, optional
        Sample frequency. Default fs=N
    P  : int, optional
        Number of periods. default = 1
    R  : int, optional
        Number of realizations. default = 1
    lines : array_like or str: {'full', 'odd', 'oddrandom'}, optional
        For characterization of NLs, only selected lines are excited.
    rms : float, optional
        rms(amplitude) of the generated signals. default = 1. Note that since
        the signal is zero-mean, the std and rms is equal.
    ngroup : int, optional, default = 4
        In case of ftype = oddrandom, 1 out of ngroup odd lines is discarded.

    Returns
    -------
    u: RxNP record of the generated signals
    lines: excited frequency lines -> 1 = dc, 2 = fs/N
    freq: frequency vector

    Examples
    --------
    Generate two realizations of a full multisine with 1000 samples and
    excitation up to one third of the Nyquist frequency.
    The two realizations have the same amplitude spectrum, but different phase
    realizations (uniformly distributed between [-π,π))
    >>> N = 1000  # One thousand samples
    >>> kind = 'full'  # Full multisine
    >>> f2 = round(N//6)  # Excitation up to one sixth of the sample frequency
    >>> R = 2   # Two phase realizations
    >>> u, lines, freq = multisine(f2=f2,N=N,lines=kind,R=R)

    Generate a random odd multisine where the excited odd lines are split in
    groups of three consecutive lines and where one line is randomly chosen in
    each group to be a detection line (i.e. without excitation)
    >>> kind = 'oddrandom'
    >>> u1,lines, freq = multisine(f2=f2,N=N,lines=kind,R=1,ngroup=3)
    Generate another phase realization of this multisine with the same excited
    lines and detection lines
    >>> u2,*_ = multisine(N=N,lines=lines,R=1)

    Notes
    -----
    J.Schoukens, M. Vaes, and R. Pintelon:
    Linear System Identification in a Nonlinear Setting:
    Nonparametric Analysis of the Nonlinear Distortions and Their Impact on the
    Best Linear Approximation. https://arxiv.org/pdf/1804.09587.pdf

    """
    if fs is None:
        fs = N
    if f2 is None:
        f2 = np.floor(0.9*N/2)
    if not fs >= 2*f2:
        raise AssertionError(f"fs should be {fs} >= {2*f2}")
    if not N >= 2*f2:
        raise AssertionError('N should be higher than Nyquist freq, '
                             'N >= 2*f2. N={}, f2={}'.format(N, f2))

    VALID_LINES = {'full', 'odd', 'oddrandom'}
    if isinstance(lines, str) and lines.lower() in VALID_LINES:
        lines = lines.lower()
        # frequency resolution
        f0 = fs/N
        # lines selection - select which frequencies to excite
        lines_min = np.ceil(f1/f0).astype('int')
        lines_max = np.floor(f2/f0).astype('int')
        _lines = np.arange(lines_min, lines_max, dtype=int)
    elif isinstance(lines, (np.ndarray, list)):  # user specified lines
        _lines = np.array(lines)
    else:
        raise ValueError(f"Invalid lines-type. Should be one of {VALID_LINES}"
                         f" or array of frequency lines. Is {lines}")

    # remove dc
    if _lines[0] == 0:
        _lines = _lines[1:]

    if isinstance(lines, str):
        if lines == 'full':
            pass  # do nothing
        elif lines == 'odd':
            # remove even lines
            if np.remainder(_lines[0], 2):  # lines[0] is even
                _lines = _lines[::2]
            else:
                _lines = _lines[1::2]
        elif lines == 'oddrandom':
            if np.remainder(_lines[0], 2):
                _lines = _lines[::2]
            else:
                _lines = _lines[1::2]
            # remove 1 out of ngroup lines
            nlines = len(_lines)
            nremove = np.floor(nlines/ngroup).astype('int')
            idx = np.random.randint(ngroup, size=nremove)
            idx = idx + ngroup*np.arange(nremove)
            _lines = np.delete(_lines, idx)

    nlines = len(_lines)
    # multisine generation - frequency domain implementation
    U = np.zeros((R, N), dtype=complex)
    # excite the selected frequencies
    U[:, _lines] = np.exp(2j*np.pi*np.random.rand(R, nlines))

    u = 2*np.real(ifft(U, axis=1))  # go to time domain
    u = rms*u / np.std(u[0])  # rescale to obtain desired rms/std

    # Because the ifft is for [0,2*pi[, there is no need to remove any point
    # when the generated signal is repeated.
    u = np.tile(u, (1, P))  # generate P periods
    freq = np.arange(N)/N*fs
    t = np.arange(N*P)/fs

    return u, _lines, freq, t


def sine(A, f=None, omega=None, t=None, fs=None, ns=None, phi_f=0):
    """Signle sine excitation

    Parameters
    ----------
    A: float
        Amplitude in N
    f: float
        Forcing frequency in (Hz/s)
    t: ndarray
        Time array
    phi_f: float
        Phase in degree
    """

    if t is None:
        t = np.arange(ns)/fs
    if f is not None:
        omega = f * 2*np.pi

    phi = phi_f / 180 * np.pi
    u = A * np.sin(omega*t + phi)

    return u, t


def toMDOF(u, ndof, fdof):
    """Convert a vector to multidof

    u: ndarray
        forcing
    ndof: int
        Number of DOFs
    fdof: int or ndarray of int
        DOF location of force(s)
    """
    fdofs = np.atleast_1d(np.asarray(fdof))
    if any(fdofs) > ndof-1:
        raise ValueError('Some fdofs are greater than system size(n-1), {}>{}'.
                         format(fdofs, ndof-1))

    ns = len(u)
    f = np.zeros((ndof, ns))
    # add force to dofs
    for dof in fdofs:
        f[dof] = f[dof] + u

    return f
