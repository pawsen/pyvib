#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import solve
from scipy.linalg import eigvals, logm, norm


def is_stable(A, domain='z'):
    """Determines if a linear state-space model is stable from eigenvalues of `A`

    Parameters
    ----------
    A : ndarray(n,n)
        state matrix
    domain : str, optional {'z', 's'}
        'z' for discrete-time, 's' for continuous-time state-space models

    returns
    -------
    bool
    """

    if domain == 'z':  # discrete-time
        # Unstable if at least one pole outside unit circle
        if any(abs(eigvals(A)) > 1):
            return False
    elif domain == 's':  # continuous-time
        # Unstable if at least one pole in right-half plane
        if any(np.real(eigvals(A)) > 0):
            return False
    else:
        raise ValueError(f"{domain} wrong. Use 's' or 'z'")
    return True

def ss2phys(A, B, C, D=None):
    """Calculate state space matrices in physical domain using a similarity
    transform T

    See eq. (20.10) in
    Etienne Gourc, JP Noel, et.al
    "Obtaining Nonlinear Frequency Responses from Broadband Testing"
    https://orbi.uliege.be/bitstream/2268/190671/1/294_gou.pdf
    """

    # Similarity transform
    T = np.vstack((C, C @ A))
    C = solve(T.T, C.T).T        # (C = C*T^-1)
    A = solve(T.T, (T @ A).T).T  # (A = T*A*T^-1)
    B = T @ B
    return A, B, C, T

def ss2frf(A, B, C, D, freq):
    """Compute frequency response function from state-space parameters
    (discrete-time)

    Computes the frequency response function (FRF) or matrix (FRM) Ĝ at the
    normalized frequencies `freq` from the state-space matrices `A`, `B`, `C`,
    and `D`. ```̂G(f) = C*inv(exp(2j*pi*f)*I - A)*B + D```

    Returns
    -------
    Gss : ndarray(F,p,m)
        frequency response matrix

    """
    # Z-transform variable
    z = np.exp(2j*np.pi*freq)
    In = np.eye(*A.shape)
    # Use broadcasting. Much faster than for loop.
    Gss = C @ solve((z*In[...,None] - A[...,None]).transpose((2,0,1)), B[None]) + D
    return Gss

def discrete2cont(ad, bd, cd, dd, dt, method='zoh', alpha=None):
    """Convert linear system from discrete to continuous time-domain.

    This is the inverse of :func:`scipy.signal.cont2discrete`. This will not
    work in general, for instance with the ZOH method when the system has
    discrete poles at ``0``.

    Parameters
    ----------
    A,B,C,D : :data:`linear_system_like`
       Linear system representation.
    dt : ``float``
       Time-step used to *undiscretize* ``sys``.
    method : ``string``, optional
       Method of discretization. Defaults to zero-order hold discretization
       (``'zoh'``), which assumes that the input signal is held constant over
       each discrete time-step.
    alpha : ``float`` or ``None``, optional
       Weighting parameter for use with ``method='gbt'``.
    Returns
    -------
    :class:`.LinearSystem`
       Continuous linear system (``analog=True``).
    See Also
    --------
    :func:`scipy.signal.cont2discrete`
    Examples
    --------
    Converting a linear system
    >>> from nengolib.signal import discrete2cont, cont2discrete
    >>> from nengolib import DoubleExp
    >>> sys = DoubleExp(0.005, 0.2)
    >>> assert dsys == discrete2cont(cont2discrete(sys, dt=0.1), dt=0.1)

    """

    sys = (ad, bd, cd, dd)
    if dt <= 0:
        raise ValueError("dt (%s) must be positive" % (dt,))

    n = ad.shape[0]
    m = n + bd.shape[1]

    if method == 'gbt':
        if alpha is None or alpha < 0 or alpha > 1:
            raise ValueError("alpha (%s) must be in range [0, 1]" % (alpha,))

        In = np.eye(n)
        ar = solve(alpha*dt*ad.T + (1-alpha)*dt*In, ad.T - In).T
        M = In - alpha*dt*ar

        br = np.dot(M, bd) / dt
        cr = np.dot(cd, M)
        dr = dd - alpha*np.dot(cr, bd)

    elif method in ('bilinear', 'tustin'):
        return discrete2cont(*sys, dt, method='gbt', alpha=0.5)

    elif method in ('euler', 'forward_diff'):
        return discrete2cont(*sys, dt, method='gbt', alpha=0.0)

    elif method == 'backward_diff':
        return discrete2cont(*sys, dt, method='gbt', alpha=1.0)

    elif method == 'zoh':
        # see https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
        M = np.zeros((m, m))
        M[:n, :n] = ad
        M[:n, n:] = bd
        M[n:, n:] = np.eye(bd.shape[1])
        E = logm(M) / dt

        ar = E[:n, :n]
        br = E[:n, n:]
        cr = cd
        dr = dd
    else:
        raise ValueError("invalid method: '%s'" % (method,))

    return ar, br, cr, dr
