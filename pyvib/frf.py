#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft
from numpy.linalg import pinv

def periodic(u, y, fs=None, fmin=None, fmax=None): # signal
    """Interface to periodic FRF

    Extract periodic signal from original signal and call periodic FRF

    Parameters
    ----------
    nper : int (optional)
        Number of periods in measurement
    """

    if fs is None:
        # fs = signal.fs
        pass

    if None in (fmin, fmax):
        # flines = signal.flines
        pass
    else:
        npp = u.shape[0]
        freq = np.arange(npp) * fs/npp
        flines = np.where((freq >= fmin) & (freq <= fmax))
        freq = freq[flines]

    # If signal is cut, used that. Otherwise use the full signal.
    # if signal.iscut:
    #     u = signal.u_per.squeeze()
    #     y = signal.y_per
    # else:
    #     u = signal.u
    #     y = signal.y

    U = fft(u, axis=0)[flines].transpose((1,2,3,0))
    Y = fft(y, axis=0)[flines].transpose((1,2,3,0))
    G, covG, covGn = bla_periodic(U, Y)
    G = G.transpose((2,0,1))
    if covG is not None:
        covG = covG.transpose((2,0,1))
    if covGn is not None:
        covGn = covGn.transpose((2,0,1))

    return freq, G, covG, covGn


def bla_periodic(U, Y):  #(u, y, nper, fs, fmin, fmax):
    """Calculate the frequency response matrix, and the corresponding noise and
    total covariance matrices from the spectra of periodic input/output data.

    Note that the term stochastic nonlinear contribution term is a bit
    misleading. The NL contribution is deterministic given the same forcing
    buy differs between realizations.

    G(f) = FRF(f) = Y(f)/U(f) (Y/F in classical notation)
    Y and U is the output and input of the system in frequency domain.

    Parameters
    ----------
    u : ndarray
        Forcing signal
    y : ndarray
        Response signal (displacements)
    fs : float
        Sampling frequency
    fmin : float
        Starting frequency in Hz
    fmax : float
        Ending frequency in Hz

    Returns
    -------
    G : ndarray
        Frequency response matrix(FRM)
    covGML : ndarray
        Total covariance (= stochastic nonlinear contributions + noise)
    covGn : ndarray
        Noise covariance
    """

    # Number of inputs, realization, periods and frequencies
    m, R, P, F = U.shape
    p = Y.shape[0]  # number of outputs
    M = np.floor(R/m).astype(int)  # number of block of experiments
    if M*m != R:
        print('Warning: suboptimal number of experiments: B*m != M')
    # Reshape in M blocks of m experiments
    U = U[:,:m*M].reshape((m,m,M,P,F))
    Y = Y[:,:m*M].reshape((p,m,M,P,F))

    if P > 1:
        # average input/output spectra over periods
        U_mean = np.mean(U,axis=3)  # m x m x M x F
        Y_mean = np.mean(Y,axis=3)

        # Estimate noise spectra
        # create new axis. We could have used U_m = np.mean(U,3, keepdims=True)
        NU = U - U_mean[:,:,:,None,:]  # m x m x M x P x F
        NY = Y - Y_mean[:,:,:,None,:]

        # Calculate input/output noise (co)variances on averaged(over periods)
        # spectra
        covU = np.empty((m*m,m*m,M,F), dtype=complex)
        covY = np.empty((p*m,p*m,M,F), dtype=complex)
        covYU = np.empty((p*m,m*m,M,F), dtype=complex)
        for mm in range(M):  # Loop over experiment blocks
            # noise spectrum of experiment block mm (m x m x P x F)
            NU_m = NU[:,:,mm]
            NY_m = NY[:,:,mm]
            for f in range(F):
                # TODO extend this using einsum, so we avoid all loops
                # TODO fx: NU_m[...,f].reshape(-1,*NU_m[...,f].shape[2:])
                # flatten the m x m dimension and use einsum to take the outer
                # product of m*m x m*m and then sum over the p periods.
                tmpUU = NU_m[...,f].reshape(-1, P)  # create view
                tmpYY = NY_m[...,f].reshape(-1, P)
                covU[:,:,mm,f] = np.einsum('ij,kj->ik',tmpUU,tmpUU.conj()) / (P-1)/P
                covY[:,:,mm,f] = np.einsum('ij,kj->ik',tmpYY,tmpYY.conj()) / (P-1)/P
                covYU[:,:,mm,f] = np.einsum('ij,kj->ik',tmpYY,tmpUU.conj()) / (P-1)/P

        # Further calculations with averaged spectra
        U = U_mean  # m x m x M x F
        Y = Y_mean
    else:
        U = U.squeeze(axis=3)
        Y = Y.squeeze(axis=3)

    # Compute FRM and noise and total covariance on averaged(over experiment
    # blocks and periods) FRM
    G = np.empty((p,m,F), dtype=complex)
    covGML = np.empty((m*p,m*p,F), dtype=complex)
    covGn = np.empty((m*p,m*p,F), dtype=complex)
    Gm = np.empty((p,m,M), dtype=complex)
    U_inv_m = np.empty((m,m,M), dtype=complex)
    covGn_m = np.empty((m*p,m*p,M), dtype=complex)

    for f in range(F):
        # Estimate the frequency response matrix (FRM)
        for mm in range(M):
            # psudo-inverse by svd. A = usvᴴ, then A⁺ = vs⁺uᴴ where s⁺=1/s
            U_inv_m[:,:,mm] = pinv(U[:,:,mm,f])
            # FRM of experiment block m at frequency f
            Gm[:,:,mm] = Y[:,:,mm,f] @ U_inv_m[:,:,mm]

        # Average FRM over experiment blocks
        G[:,:,f] = Gm.mean(2)

        # Estimate the total covariance on averaged FRM
        if M > 1:
            NG = G[:,:,f,None] - Gm
            tmp = NG.reshape(-1, M)
            covGML[:,:,f] = np.einsum('ij,kj->ik',tmp,tmp.conj()) / M/(M-1)

        # Estimate noise covariance on averaged FRM (only if P > 1)
        if P > 1:
            for mm in range(M):
                U_invT = U_inv_m[:,:,mm].T
                A = np.kron(U_invT, np.eye(p))
                B = -np.kron(U_invT, Gm[:,:,mm])
                AB = A @ covYU[:,:,mm,f] @ B.conj().T
                covGn_m[:,:,mm] = A @ covY[:,:,mm,f] @ A.conj().T + \
                    B @ covU[:,:,mm,f] @ B.conj().T + \
                    (AB + AB.conj().T)

            covGn[:,:,f] = covGn_m.mean(2)/M

    # No total covariance estimate possible if only one experiment block
    if M < 2:
        covGML = None
    # No noise covariance estimate possible if only one period
    if P == 1:
        covGn = None

    return G, covGML, covGn

def covariance(y):
    """Compute covariance matrix output spectra due to noise from signal y

    The variation is calculated along the periods and averaged over the
    realizations.

    Parameters
    ----------
    y : ndarray(npp,p,R,P)
        signal where npp is the number of points per period, p is the number of
        dofs, R is the number of  realizations, and P is the number of
        periods

    Returns
    -------
    covY : ndarray(nfd,p,p)
        covariance matrix of the dof(s).
    """

    # Number of samples, outputs, realizations, and periods
    npp,p,R,P = y.shape
    Y = fft(y, axis=0)
    # Number of bins in positive half of the spectrum
    nfd = int(npp/2)
    Y = Y[:nfd]
    # average over periods
    Ymean = Y.mean(3)
    # Variations over periods
    NY = Y - Ymean[...,None]  # (npp,p,R,P)
    # TODO fix NY
    #NY = permute(NY,[2 3 4 1]); % p x R x P x NFD
    covY = np.empty((p,p,R,nfd), dtype=complex)
    for f in range(nfd):
        for r in range(R):
            tmpYY = NY[f,:,r,:].reshape(-1, P)
            covY[...,r,f] = np.einsum('ij,kj->ik',tmpYY,tmpYY.conj()) / (P-1)/P

    # average over all realizations
    covY = covY.mean(2)
    covY = covY.T
    return covY

def nonperiodic(u, y, N, fs, fmin, fmax):
    """Calculate FRF for a nonperiodic signal.

    A nonperiodic signal could be a hammer test. For nonperiodid signals,
    normally H1 or H2 is calculated. H2 is most commonly used with random
    excitation.
    H1 is used when the output is expected to be noisy compared to the input.
    H2 is used when the input is expected to be noisy compared to the output.

    H1 = Suy/Suu
    H2 = Syy/Syu

    All spectral densities are calculated in frequency domain
    Suy is the Cross Spectral Density of the input and output
    Suu/Syy is the Auto Spectral Density of the input/output. Also called power
        spectral density(PSD)
    Suy = Syu* (complex conjugate). See [1]_


    Parameters
    ----------
    u : ndarray
        Forcing signal
    y : ndarray
        Response signal (displacements)
    N : int
        Number of points used for fft. N =  2^nfft, where nfft standard is
        chosen as 8.
    fs : float
        Sampling frequency
    fmin : float
        Starting frequency in Hz
    fmax : float
        Ending frequency in Hz

    Returns
    -------
    freq : ndarray
        Frequencies for the FRF
    FRF : ndarray
        Receptance vector. Also called H.
    sigT : ndarray

    gamma : ndarray
        Coherence. Between 0 and 1 that measures the correlation between u(n)
        and y(n) at the frequency f, ie. can y be predicted from u. The
        coherence of a linear system therefore represents the fractional part
        of the output signal power that is produced by the input at that
        frequency. 1 is perfect correlation, 0 is none.

    Notes -----
    [1]: Ewins, D. J. "Modal testing: theory, practice and application (2003),
    pages 141.
    https://en.wikipedia.org/wiki/Spectral_density
    https://en.wikipedia.org/wiki/Coherence_(signal_processing)

    """

    while N > len(u)/2:
        N = int(N//2)

    M = int(np.floor(len(u)/N))

    freq = (np.arange(1, N/2 - 1) + 0.5) * fs/N
    flines = np.where((freq >= fmin) & (freq <= fmax))
    freq = freq[flines]
    nfreq = len(freq)

    u = u[:M * N]
    y = y[:M * N]

    u = np.reshape(u, (N, M))
    y = np.reshape(y, (N, M))

    U = np.fft.fft(u, axis=0) / np.sqrt(N)
    U = U[1:N/2+1,:]
    Y = np.fft.fft(y, axis=0) / np.sqrt(N)
    Y = Y[1:N/2+1,:]

    U = np.diff(U, axis=0)
    Y = np.diff(Y, axis=0)

    # Syu: Cross spectral density
    # Suu: power spectral density
    # (mayby not correct: taking the mean, is the same as taking the expectance
    # E)
    Syu = np.mean(Y * U.conj(), axis=1)
    Suu = np.mean(np.abs(U)**2, axis=1)
    Syy = np.mean(np.abs(Y)**2, axis=1)
    FRF = Syu / Suu
    FRF = FRF[flines]

    if M > 1:
        sigT = 1/(M-1) * (Syy - np.abs(Syu)**2/Suu) / Suu
        # coherence
        gamma = np.abs(Syu)**2 / (Suu*Syy)
        sigT = sigT[flines]
        gamma = gamma[flines]
    else:
        sigT = None
        gamma = None

    return freq, FRF, sigT, gamma
