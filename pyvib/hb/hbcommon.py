#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.fftpack import fft, ifft

def hb_components(z, n, NH):
    """Get HB coefficient c's

    Parameters
    ----------
    n : int
        Number of DOFs, ie M.shape[0]
    """
    z = np.hstack([np.zeros(n), z])
    # reorder so first column is zeros, then one column for each dof
    z = np.reshape(z, (n,2*(NH+1)), order='F')

    # first column in z is zero, thus this will a 'division by zero' warning.
    # Instead we just set the first column in phi to pi/2 (arctan(inf) = pi/2)
    # phi = np.arctan(z[:,1::2] / z[:,::2])
    phi = np.empty((n, NH+1))
    phi[:,1:] = np.arctan(z[:,3::2] / z[:,2::2])
    phi[:,0] = np.pi/2

    c = z[:,::2] / np.cos(phi)
    c[:,0] = z[:,1]

    # normalize components for each dof
    cnorm = np.abs(c)/np.sum(np.abs(c),axis=1)[:,None]

    return c, phi, cnorm

def hb_signal(omega, t, c, phi):
    """Get real signal from HB coefficients(components)"""
    n = c.shape[0]
    NH = c.shape[1]-1
    nt = len(t)
    tt = np.arange(1,NH+1)[:,None] * omega * t

    x = np.zeros((n, nt))
    for i in range(n):

        tmp = tt + np.outer(phi[i,1:],np.ones(nt))
        tmp = c[i,0]*np.ones(nt) + c[i,1:] @ np.sin(tmp)
        x[i] = tmp  # np.sum(tmp, axis=0)

    return x

def fft_coeff(x, NH):
    """ Extract FFT-coefficients from X=fft(x)
    """
    # n: dofs
    n, nt = x.shape
    # Format of X after transpose: (nt, n)
    X = fft(x).T / nt

    re_fft_im_fft = np.hstack([-2*np.imag(X[1:NH+1]),
                               2* np.real(X[1:NH+1])])

    # X[0] only contains real numbers (it is the dc/0-frequency-part), but we
    # still need to extract the real part. Otherwise z is casted to complex128
    z = np.hstack([np.real(X[0]), re_fft_im_fft.ravel()])

    return z

def ifft_coeff(z, n, nt, NH):
    """ extract iFFT-coefficients from x=ifft(X)
    """

    X = np.zeros((n, nt), dtype='complex')
    X[:,0] = nt*z[:n]

    for i in range(NH):
        X[:,i+1] = 2 * (nt/2   * z[(n*(2*i+1)+n): (n*(2*i+1)+2*n)] - \
                        nt/2*1j* z[(n*(2*i+1))  : (n*(2*i+1)+n)])

    x = np.real(ifft(X))

    return x
