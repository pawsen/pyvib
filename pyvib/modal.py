#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" LSCE, LSCF, LSFD are modified from OpenModal
https://github.com/openmodal/
Copyright (C) 2014-2017 Matjaž Mršnik, Miha Pirnat, Janko Slavič, Blaž Starc
(in alphabetic order)

The rest is by
Paw Møller <pawsen@gmail.com>
"""

import numpy as np
from numpy.fft import irfft
from scipy.linalg import lstsq, toeplitz, eig, inv, norm, solve
from collections import defaultdict
from .common import window


def lsce(frf, f, low_lim, nmax, fs, additional_timepoints=0):
    """Compute poles(natural frequencies and damping) from FRFs.

    The Least-Squares Complex Exponential method (LSCE), introduced in [1]_, is
    the extension of the Complex Exponential method (CE) to a global procedure.
    It is therefore a SIMO method, processing simultaneously several IRFs
    obtained by exciting a structure at one single point and measuring the
    responses at several locations. With such a procedure, a consistent set of
    global parameters (natural frequencies and damping factors) is obtained,
    thus overcoming the variations obtained in the results for those parameters
    when applying the CE method on different IRFs.

    The output from LSCE is used by LSFD to compute mode shapes.

    Parameters
    ----------
    frf: ndarray
        frequency response function array - receptance
    f: float
        starting frequency
    low_lim: float
        lower limit of the frf/f
    nmax: int
        the maximal order of the polynomial
    fs: float
        time sampling interval
    additional_timepoints: float, default 0
        normed additional time points (default is 0% added time points, max. is
        1, all time points (100%) taken into computation)

    Returns
    -------
    srlist: list
        list of complex eigenfrequencies

    References
    -----------
    [1] Brown, D. L., Allemang, R. J. Zimmermann, R., Mergeay, M.,
        "Parameter Estimation Techniques For Modal Analysis"
        SAE Technical Paper Series, No. 790221, 1979
    [2] Ewins, D.J .; Modal Testing: Theory, practice and application,
        second edition. Reasearch Studies Press, John Wiley & Sons, 2000.
    [3] N. M. M. Maia, J. M. M. Silva, J. He, N. A. J. Lieven, R. M.
        Lin, G. W. Skingle, W. To, and A. P. V Urgueira. Theoretical
        and Experimental Modal Analysis. Reasearch Studio Press
        Ltd., 1997.
    [4] Kerschen, G., Golinval, J.-C., Experimental Modal Analysis,
        http://www.ltas-vis.ulg.ac.be/cmsms/uploads/File/Mvibr_notes.pdf

    """

    # number of outputs, length of receptance
    no, l = frf.shape
    # number of DFT frequencies (nf >> n)
    nf = 2*(l-low_lim-1)

    # Impulse response function, ie. h = IFFT(H)
    irf = np.fft.irfft(frf[:, low_lim:], n=nf, axis=-1)

    sr_list = []
    nf2 = irf.shape[1]
    for n in range(1, nmax+1):
        # number of time points for computation
        nt = int(2*n + additional_timepoints*(nf2 - 4*n))

        # setup equation system.
        # [h]: time-response matrix, hh: {h'} vector, size (2N)x1
        h = np.zeros((nt*no, 2*n))
        hh = np.zeros(nt*no)

        for j in range(nt):
            for k in range(no):
                h[j+k*2*n, :] = irf[k, j:j+2*n]
                hh[j+k*2*n] = irf[k, (2*n)+j]

        # the computation of the autoregressive coefficients matrix
        beta = lstsq(h, -hh)[0]
        sr = np.roots(np.append(beta, 1)[::-1])  # the roots of the polynomial
        sr = (np.log(sr)*fs).astype(complex)     # the complex natural frequency
        sr += 2*np.pi*f*1j  # for f_min different than 0 Hz
        # sort after eigenvalues
        sr_list.append(sr.sort())

    return sr_list

def lsce_reconstruction(n, f, sr, vr, irf, two_sided_frf=False):
    """Reconstruction of the least-squares complex exponential (CE) method.

    :param n: number of degrees of freedom
    :param f: frequency vector [Hz]
    :param sr: the complex natural frequency
    :param vr: the roots of the polynomial
    :param irf: impulse response function vector
    :return: residues and reconstructed FRFs
    """
    dt = 1/(len(f)*(f[1]-f[0]))
    if two_sided_frf is False:
        dt /= 2

    # no: number of outputs
    no, l = irf.shape
    v = np.zeros((2*n, 2*n), dtype=complex)
    for l in range(0, 2*n):
        for k in range(0, 2*n):
            v[k, l] = vr[l]**k

    # {h''} vector
    hhh = np.zeros((2*n*no))
    for j in range(0, 2*n):
        for k in range(no):
            hhh[j+k*2*n] = irf[k, j]

    a = np.zeros((no, 2*n), dtype=complex)
    for i in range(no):
        # the computation of residues
        a[i, :] = np.linalg.solve(v, -hhh[i*2*n:(i+1)*2*n])

    # reconstructed irf
    h = np.zeros(np.shape(irf))

    for i in range(no):
        for jk in range(l):
            h[i, jk] = np.real(np.sum(a[i,:]*np.exp(sr*jk*dt)))

    return a, h


def lsfd(lambdak, f, frf):
    """LSFD (Least-Squares Frequency domain) method

    Determine the residues and mode shapes from complex natural frquencies and
    the measured frequency response functions.

    Parameters
    ----------
    lambdak: ndarray
        a vector of selected complex natural frequencies
    f: ndarray
        frequency vector
    frf: ndarray
        frequency response functions

    Returns
    -------
    h, a, lr, ur
        reconstructed FRF, modal constant(residue), lower residual,
        upper residual
    """

    ni = frf.shape[0]  # number of references
    no = frf.shape[1]  # number of responses
    n = frf.shape[2]   # length of frequency vector
    nmodes = lambdak.shape[0]  # number of modes

    omega = 2 * np.pi * f  # angular frequency

    # Factors in the freqeuncy response function
    b = 1 / np.subtract.outer(1j * omega, lambdak).T
    c = 1 / np.subtract.outer(1j * omega, np.conj(lambdak)).T

    # Separate complex data to real and imaginary part
    hr = frf.real
    hi = frf.imag
    br = b.real
    bi = b.imag
    cr = c.real
    ci = c.imag

    # Stack the data together in order to obtain 2D matrix
    hri = np.dstack((hr, hi))
    bri = np.hstack((br+cr, bi+ci))
    cri = np.hstack((-bi+ci, br-cr))

    ur_multiplyer = np.ones(n)
    ur_zeros = np.zeros(n)
    lr_multiplyer = -1/(omega**2)

    urr = np.hstack((ur_multiplyer, ur_zeros))
    uri = np.hstack((ur_zeros, ur_multiplyer))
    lrr = np.hstack((lr_multiplyer, ur_zeros))
    lri = np.hstack((ur_zeros, lr_multiplyer))

    bcri = np.vstack((bri, cri, urr, uri, lrr, lri))

    # Reshape 3D array to 2D for least squares coputation
    hri = hri.reshape(ni*no, 2*n)

    # Compute the modal constants (residuals) and upper and lower residuals
    uv = lstsq(bcri.T,hri.T)[0]

    # Reshape 2D results to 3D
    uv = uv.T.reshape(ni, no, 2*nmodes+4)

    u = uv[:, :, :nmodes]
    v = uv[:, :, nmodes:-4]
    urr = uv[:, :, -4]
    uri = uv[:, :, -3]
    lrr = uv[:, :, -2]
    lri = uv[:, :, -1]

    a = u + 1j*v  # Modal constant (residue)
    ur = urr + 1j*uri  # Upper residual
    lr = lrr + 1j*lri  # Lower residual

    # Reconstructed FRF matrix
    h = uv @ bcri
    h = h[:,:,:n] + 1j*h[:,:,n:]

    return h, a, lr, ur

def lscf(frf, low_lim, n, fs):
    """LSCF - Least-Squares Complex frequency domain method

    The LSCF method is an frequency-domain Linear Least Squares estimator
    optimized for modal parameter estimation. The choice of the most important
    algorithm characteristics is based on the results in [1] (Section 5.3.3.)
    and can be summarized as:

    - Formulation: the normal equations [1]_

    (Eq. 5.26: [sum(Tk - Sk.H * Rk^-1 * Sk)]*ThetaA=D*ThetaA = 0) are
    constructed for the common denominator discrete-time model in the Z-domain.
    Consequently, by looping over the outputs and inputs, the submatrices Rk,
    Sk, and Tk are formulated through the use of the FFT algorithm as Toeplitz
    structured (n+1) square matrices. Using complex coefficients, the FRF data
    within the frequency band of interest (FRF-zoom) is projected in the
    Z-domain in the interval of [0, 2*pi] in order to improve numerical
    conditioning. (In the case that real coefficients are used, the data is
    projected in the interval of [0, pi].) The projecting on an interval that
    does not completely describe the unity circle, say [0, alpha*2*pi] where
    alpha is typically 0.9-0.95. Deliberately over-modeling is best applied to
    cope with discontinuities. This is justified by the use of a discrete time
    model in the Z-domain, which is much more robust for a high order of the
    transfer function polynomials.

    - Solver: the normal equations can be solved for the denominator
    coefficients ThetaA by computing the Least-Squares (LS) or mixed
    Total-Least-Squares (TLS) solution. The inverse of the square matrix D for
    the LS solution is computed by means of a pseudo inverse operation for
    reasons of numerical stability, while the mixed LS-TLS solution is computed
    using an SVD (Singular Value Decomposition).

    Parameters
    ----------
    frf: ndarray
        frequency response function - receptance
    low_lim:
        lower limit of the frf
    n: int
        the order of the polynomial
    fs: float
        time sampling interval

    Returns
    -------
    srlist: list
        list of complex eigenfrequencies

    References
    ----------
    [1] Verboven, P., Frequency-domain System Identification for Modal
        Analysis, Ph. D. thesis, Mechanical Engineering Dept. (WERK), Vrije
        Universiteit Brussel, Brussel, (Belgium), May 2002,
        (http://mech.vub.ac.be/avrg/PhD/thesis_PV_web.pdf)

    [2] Verboven, P., Guillaume, P., Cauberghe, B., Parloo, E. and Vanlanduit
        S., Stabilization Charts and Uncertainty Bounds For Frequency-Domain
        Linear Least Squares Estimators, Vrije Universiteit Brussel(VUB),
        Mechanical Engineering Dept. (WERK), Acoustic and Vibration Research
        Group (AVRG), Pleinlaan 2, B-1050 Brussels, Belgium, e-mail:
        Peter.Verboven@vub.ac.be, url:
        (http://sem-proceedings.com/21i/sem.org-IMAC-XXI-Conf-s02p01
        -Stabilization-Charts-Uncertainty-Bounds-Frequency-Domain-
        Linear-Least.pdf)

    [3] P. Guillaume, P. Verboven, S. Vanlanduit, H. Van der Auweraer, B.
        Peeters, A Poly-Reference Implementation of the Least-Squares Complex
        Frequency-Domain Estimator, Vrije Universiteit Brussel, LMS
        International

    """

    # the poles should be complex conjugate, thus expect even polynomial order
    n *= 2

    # nr: (number of inputs) * (number of outputs), l: length of receptance
    nr, l = frf.shape

    # number of DFT frequencies (nf >> n)
    nf = 2*(l-1)

    indices_s = np.arange(-n, n+1)
    indices_t = np.arange(n+1)

    # Selection of the weighting function
    # Least-Squares (LS) Formulation based on Normal Matrix
    sk = -irfft_adjusted_lower_limit(frf, low_lim, indices_s)
    t = irfft_adjusted_lower_limit(frf.real**2 + frf.imag**2,
                                   low_lim, indices_t)
    r = -(irfft(np.ones(low_lim), n=nf))[indices_t]*nf
    r[0] += nf

    s = []
    for i in range(nr):
        s.append(toeplitz(sk[i, n:], sk[i, :n+1][::-1]))
    t = toeplitz(np.sum(t[:, :n+1], axis=0))
    r = toeplitz(r)

    sr_list = []
    for j in range(2, n+1, 2):
        d = 0
        for i in range(nr):
            rinv = inv(r[:j+1, :j+1])
            snew = s[i][:j+1, :j+1]
            # sum
            d -= (snew[:j+1, :j+1].T @ rinv) @ snew[:j+1, :j+1]
        d += t[:j+1, :j+1]

        a0an1 = solve(-d[0:j, 0:j], d[0:j, j])
        # the numerator coefficients
        sr = np.roots(np.append(a0an1, 1)[::-1])
        # Z-domain (for discrete-time domain model)
        sr = -np.log(sr) * fs
        sr_list.append(sr.sort())

    return sr_list

def remove_redundant(omega, xi, prec=1e-3):
    """Remove the redundant values of frequency and damping vectors
    (due to the complex conjugate eigenvalues)

    Input:
    omega - eiqenfrquencies vector
    xi - damping ratios vector
    prec - absoulute precision in order to distinguish between two values

    """
    N = len(omega)
    test_omega = np.zeros((N,N), dtype=int)
    for i in range(1,N):
        for j in range(0,i):
            if np.abs((omega[i] - omega[j])) < prec:
                test_omega[i,j] = 1
            else:
                test_omega[i,j] = 0
    test = np.zeros(N, dtype=int)
    for i in range(0,N):
        test[i] = np.sum(test_omega[i,:])

    omega_mod = omega[np.where(test < 1)]
    xi_mod = xi[np.where(test < 1)]
    return omega_mod, xi_mod

def irfft_adjusted_lower_limit(x, low_lim, indices):
    """
    Compute the ifft of real matrix x with adjusted summation limits:

        y(j) = sum[k=-n-2, ... , -low_lim-1, low_lim, low_lim+1, ... n-2,
                   n-1] x[k] * exp(sqrt(-1)*j*k* 2*pi/n),
        j =-n-2, ..., -low_limit-1, low_limit, low_limit+1, ... n-2, n-1

    :param x: Single-sided real array to Fourier transform.
    :param low_lim: lower limit index of the array x.
    :param indices: list of indices of interest
    :return: Fourier transformed two-sided array x with adjusted lower limit.
             Retruns values.
    """

    nf = 2 * (x.shape[1] - 1)
    a = (irfft(x, n=nf)[:, indices]) * nf
    b = (irfft(x[:, :low_lim], n=nf)[:, indices]) * nf
    return a - b

def stabilization(sd, fmin=0, fmax=np.inf, tol_freq=1, tol_damping=5,
                  tol_mode=0.98, macchoice='complex'):
    """Calculate stabilization of modal parameters for increasing model order.
    Used for plotting stabilization diagram

    Parameters
    ----------
    sd: dict with keys {'wn', 'zeta', 'realmode'/'cpxmode', 'stable'}
        dict of dicts having modal parameters for each model order.
    fmin: float, default 0
        Minimum frequency to consider
    fmax: float, default np.inf
        Maximum frequency to consider
    tol_freq: float, default 1
        Tolerance for frequency in %, lower is better. Between [0, 100]
    tol_damping: float, default 5
        Tolerance for damping in %, lower is better. Between [0, 100]
    tol_freq: float, default 0.98
        Tolerance for mode shape, higher is better. Between [0, 1]
    macchoice: str, {'complex', 'real', 'None'}
        Method for comparing mode shapes. 'None' for no comparison.

    Returns
    -------
    SDout: two nested defaultdicts.
        First Keys is model order, second key is
        modal property: {stab, freq, zeta, mode} = {True, False}

    """

    # Initialize SDout as 2 nested defaultdict
    SDout = defaultdict(lambda: defaultdict(list))
    # loop over model orders except the last.
    for n, nnext in window(sd, 2):
        val = sd[n]
        # is A stable?
        SDout[n]['a_stable'].append(val['stable'])
        # loop over frequencies for current model order
        for ifr, natfreq in enumerate(val['wn']):
            if natfreq < fmin or natfreq > fmax:
                continue

            SDout[n]['freq'].append(natfreq)
            # compare with frequencies from one model order higher.
            nfreq = sd[nnext]['wn']
            tol_low = (1 - tol_freq / 100) * natfreq
            tol_high = (1 + tol_freq / 100) * natfreq
            ifreqS, = np.where((nfreq >= tol_low) & (nfreq <= tol_high))
            if ifreqS.size == 0:  # ifreqS is empty
                # the current natfreq is not stabilized
                SDout[n]['stab'].append(False)
                SDout[n]['zeta'].append(False)
                SDout[n]['mode'].append(False)
            else:
                # Stabilized in natfreq
                SDout[n]['stab'].append(True)
                # Only in very rare cases, ie multiple natfreqs are very
                # close, is len(ifreqS) != 1
                for ii in ifreqS:
                    nep = sd[nnext]['zeta'][ii]
                    ep = val['zeta'][ifr]
                    tol_low = (1 - tol_damping / 100) * ep
                    tol_high = (1 + tol_damping / 100) * ep

                    iepS, = np.where((nep >= tol_low) & (nep <= tol_high))
                    if iepS.size == 0:
                        SDout[n]['zeta'].append(False)
                    else:
                        SDout[n]['zeta'].append(True)
                if macchoice == 'complex':
                    m1 = val['cpxmode'][ifr]
                    m2 = sd[nnext]['cpxmode'][ifreqS]
                    MAC = ModalACX(m1, m2)
                elif macchoice == 'real':
                    m1 = sd[n]['realmode'][ifr]
                    m2 = sd[nnext]['realmode'][ifreqS]
                    MAC = ModalAC(m1, m2)
                else:
                    MAC = 0
                if np.max(MAC) >= tol_mode:
                    SDout[n]['mode'].append(True)
                else:
                    SDout[n]['mode'].append(False)

    return SDout

def frf_mkc(M, K, fmin, fmax, fres, C=None, idof=None, odof=None):
    """Compute the frequency response for a FEM model, given a range of
    frequencies.

    Parameters
    ----------
    M: array
        Mass matrix
    K: array
        Stiffness matrix
    C: array, optional
        Damping matrix
    fmin: float
        Minimum frequency used
    fmax: float
        Maximum frequency used
    fres: float
        Frequency resolution
    idof: array[int], default None
        Array of in dofs/modes to use. If None, use all.
    odof: array[int], default None
        Array of out dofs/modes to use. If None, use all.

    Returns
    -------
    freq: ndarray
        The frequencies where H is calculated.
    H: ndarray, [idof, odof, len(freq)]
        The transfer function. H[0,0] gives H1 for DOF1, etc.

    Examples
    --------
    >>> M = np.array([[1, 0],
    ...               [0, 1]])
    >>> K = np.array([[2, -1],
    ...               [-1, 6]])
    >>> C = np.array([[0.3, -0.02],
    ...               [-0.02, 0.1]])
    >>> freq, H = frf_mkc(M, K,  C)
    """

    n, n = M.shape
    if C is None:
        C = np.zeros(M.shape)
    # in/out DOFs to use
    if idof is None:
        idof = np.arange(n)
    if odof is None:
        odof = np.arange(n)
    n1 = len(idof)
    n2 = len(odof)

    # Create state space system, A, B, C, D. D=0
    Z = np.zeros((n, n))
    I = np.eye(n)
    A = np.vstack((
        np.hstack((Z, I)),
        np.hstack((-solve(M, K, assume_a='pos'),
                   -solve(M, C, assume_a='pos')))))
    B = np.vstack((Z, inv(M)))
    C = np.hstack((I, Z))

    F = int(np.ceil((fmax-fmin) / fres))
    freq = np.linspace(fmin, fmax, F+1)  # + F*fres

    mat = np.zeros((n1,n2,F+1), dtype=complex)
    for k in range(F+1):
        mat[...,k] = solve(((1j*2*np.pi*freq[k] * np.eye(2*n) - A)).T,
                           C[odof].T).T @ B[:,idof]

    # Map to right index.
    H = np.zeros((n1,n2,F+1), dtype=complex)
    for i in range(n2):
        il = odof[i]
        for j in range(n1):
            ic = odof[j]
            H[il,ic] = np.squeeze(mat[i,j,:]).T

    return freq, H

def modal_mkc(M, K, C=None, neigs=6):
    """Calculate natural frequencies, damping ratios and mode shapes.

    If the dampind matrix C is none or if the damping is proportional,
    wd and zeta are None.

    Parameters
    ----------
    M: array
        Mass matrix
    K: array
        Stiffness matrix
    C: array
        Damping matrix
    neigs: int, optional
        Number of eigenvalues to calculate

    Returns
    -------
    sd: dict
        dict with modal parameters.
        Keys: {'wn', 'wd', 'zeta', 'cpxmode','realmode', 'realmode'}

    Examples
    --------
    >>> M = np.array([[1, 0],
    ...               [0, 1]])
    >>> K = np.array([[2, -1],
    ...               [-1, 6]])
    >>> C = np.array([[0.3, -0.02],
    ...               [-0.02, 0.1]])
    >>> sd = modes_system(M, K, C)
    """

    # Damping is non-proportional, eigenvectors are complex.
    if (C is not None and not np.all(C == 0)):
        n = len(M)
        Z = np.zeros((n, n))
        I = np.eye(n)
        # creates state space matrices
        A = np.vstack([np.hstack([Z, I]),
                       np.hstack([-solve(M, K, assume_a='pos'),
                                  -solve(M, C, assume_a='pos')])])
        C = np.hstack((I, Z))
        sd = modal_ac(A, C)
        return sd

    # Damping is proportional or zero, eigenvectors are real
    egval, egvec = eig(K,M)
    lda = np.real(egval)
    idx = np.argsort(lda)
    lda = lda[idx]
    # In Hz
    wn = np.sqrt(lda) / (2*np.pi)
    realmode = np.real(egvec.T[idx])
    # normalize realmode
    nmodes = realmode.shape[0]
    for i in range(nmodes):
        realmode[i] = realmode[i] / norm(realmode[i])
        if realmode[i,0] < 0:
            realmode[i] = -realmode[i]

    zeta = []
    cpxmode = []
    wd = []
    sd = {
        'wn': wn,
        'wd': wd,
        'zeta': zeta,
        'cpxmode': cpxmode,
        'realmode': realmode,
    }
    return sd

def modal_ac(A, C=None):
    """Calculate eigenvalues and modes from state space matrices A and C

    Parameters
    ----------
    A, C
        State space matrices

    Returns
    -------
    sd : dict
        Keys are the names written below.

    wn: real ndarray. (modes)
        Natural frequency (Hz)
    wd: real ndarray. (modes)
        Damped frequency (Hz)
    zeta: real ndarray. (modes)
        Damping factor
    cpxmode : complex ndarray. (modes, nodes)
        Complex mode(s) shape
    realmode : real ndarray. (nodes, nodes)
        Real part of cpxmode. Normalized to 1.
    """
    from copy import deepcopy
    egval, egvec = eig(A)
    lda = egval
    # throw away very small values. Note this only works for state-space
    # systems including damping. For undamped system, imag(lda) == 0!
    idx1 = np.where(np.imag(lda) > 1e-8)
    lda = lda[idx1]
    # sort after eigenvalues
    idx2 = np.argsort(np.imag(lda))
    lda = lda[idx2]
    wd = np.imag(lda) / (2*np.pi)
    wn = np.abs(lda) / (2*np.pi)

    # Definition: np.sqrt(1 - (freq/natfreq)**2)
    zeta = - np.real(lda) / np.abs(lda)

    # cannot calculate mode shapes if C is not given
    if C is not None:

        # Transpose so cpxmode has format: (modes, nodes)
        cpxmode = (C @ egvec).T
        cpxmode = cpxmode[idx1][idx2]
        # np.real returns a view. Thus scaling realmode, will also scale the
        # part of cpxmode that is part of the view (ie the real part)
        realmode = deepcopy(np.real(cpxmode))
    else:
        cpxmode = []
        realmode = egvec[idx1][idx2]

    # normalize realmode
    nmodes = realmode.shape[0]
    for i in range(nmodes):
        realmode[i] = realmode[i] / norm(realmode[i])
        if realmode[i,0] < 0:
            realmode[i] = -realmode[i]

    sd = {
        'wn': wn,
        'wd': wd,
        'zeta': zeta,
        'cpxmode': cpxmode,
        'realmode': realmode,
    }
    return sd

def _complex_freq_to_freq_and_damp(lda):
    # get eigenfrequencies and damping rations from eigenvalues

    # throw away very small values. Note this only works for state-space
    # systems including damping. For undamped system, imag(lda) == 0!
    idx1 = np.where(np.imag(lda) > 1e-8)
    lda = lda[idx1]
    # sort after eigenvalues
    idx2 = np.argsort(np.imag(lda))
    lda = lda[idx2]
    wd = np.imag(lda) / (2*np.pi)
    wn = np.abs(lda) / (2*np.pi)

    # Definition: np.sqrt(1 - (freq/natfreq)**2)
    zeta = - np.real(lda) / np.abs(lda)

    return wn, wd, zeta

def ModalAC(M1, M2):
    """Calculate MAC value for real valued mode shapes

    M1 and M2 can be 1D arrays. Then they are recast to 2D.

    Parameters
    ----------
    M1 : ndarray (modes, nodes)
    M1 : ndarray (modes, nodes)

    Returns
    -------
    MAC : ndarray float (modes_m1, modes_m2)
        MAC value in range [0-1]. 1 is perfect fit.
    """
    if M1.ndim != 2:
        M1 = M1.reshape(-1,M1.shape[0])
    if M2.ndim != 2:
        M2 = M2.reshape(-1,M2.shape[0])

    nmodes1 = M1.shape[0]
    nmodes2 = M2.shape[0]

    MAC = np.zeros((nmodes1, nmodes2))

    for i in range(nmodes1):
        for j in range(nmodes2):
            num = M1[i].dot(M2[j])
            den = norm(M1[i]) * norm(M2[j])
            MAC[i,j] = (num/den)**2

    return MAC

def ModalACX(M1, M2):
    """Calculate MACX value for complex valued mode shapes

    M1 and M2 can be 1D arrays. Then they are recast to 2D.

    Parameters
    ----------
    M1 : ndarray (modes, nodes)
    M1 : ndarray (modes, nodes)

    Returns
    -------
    MACX : ndarray float (modes_m1, modes_m2)
        MAC value in range [0-1]. 1 is perfect fit.
    """
    if M1.ndim != 2:
        M1 = M1.reshape(-1,M1.shape[0])
    if M2.ndim != 2:
        M2 = M2.reshape(-1,M2.shape[0])

    nmodes1 = M1.shape[0]
    nmodes2 = M2.shape[0]

    MACX = np.zeros((nmodes1, nmodes2))
    for i in range(nmodes1):
        for j in range(nmodes2):
            num = (np.abs(np.vdot(M1[i],M2[j])) + np.abs(M1[i] @ M2[j]))**2
            den = np.real(np.vdot(M1[i],M1[i]) + np.abs(M1[i] @ M1[i])) * \
                  np.real(np.vdot(M2[j],M2[j]) + np.abs(M2[j] @ M2[j]))

            MACX[i,j] = num / den

    return MACX

class EMA():
    """Experimental modal analysis

    Methods:
    LSCE, Least-Squares Complex Exponential
    LSCF, Least-Squares Complex frequency. Also known as PolyMax

    """
    methods = {'lsce': lsce, 'lscf': lscf}

    def __init__(self, method, frf):
        """

        Parameters
        ----------
        method: str, {'lsce', 'lscf'}
            Used method
        """

        self.frf = frf
        try:
            self.method = self.methods[method]
        except KeyError:
            raise ValueError('invalid method. Should be one of: {}'.
                             format(self.methods.keys()))

    def id(self, *args, **kwargs):

        # get poles(eigenvalues) up till given model order
        self.lda = self.method(args, kwargs)
        return self.lda

    def modal(self):
        # get modal properties
        sr = []
        for pole in self.lda:
            fn, _, zeta = _complex_freq_to_freq_and_damp(pole)
            fn, zeta = remove_redundant(fn, zeta, 1e-3)
            sr.append({'wn':fn,
                       'zeta': zeta,
                       'cpxmode': None,
                       'realmode': None,
            })
        self.sr = sr
        return sr

    def stabilization(self, fmin=0, fmax=np.inf, tol_freq=1, tol_damping=5,
                      tol_mode=0.98, macchoice='None'):

        nmax = len(self.sda)
        nlist = np.arange(1,nmax+1)*2
        self.sd = stabilization(self.sd, nlist, fmin, fmax, tol_freq,
                                tol_damping, tol_mode, macchoice)

        return self.sd
