#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft
from scipy.interpolate import interp1d

from .common import mmul_weight
from .polynomial import multEdwdx, nl_terms, poly_deriv, combinations, select_active
from .statespace import NonlinearStateSpace, StateSpaceIdent


"""
PNLSS -- a collection of classes and functions for modeling nonlinear
linear state space systems.
"""


class PNLSS(NonlinearStateSpace, StateSpaceIdent):
    def __init__(self, *system, **kwargs):
        if len(system) == 1:  # and isinstance(system[0], StateSpace):
            sys = system
            self.signal = system[0].signal
            kwargs['dt'] = 1/self.signal.fs
        elif len(system) == 2:
            sys = system[0]
            self.signal = system[1]
            kwargs['dt'] = 1/self.signal.fs
        else:
            sys = system

        super().__init__(*sys, **kwargs)
        self.xpowers = np.empty(shape=(0, self.m+self.n))
        self.ypowers = np.empty(shape=(0, self.m+self.n))
        self.xactive = np.array([], dtype=int)
        self.yactive = np.array([], dtype=int)
        self.n_nx = len(self.xactive)
        self.n_ny = len(self.yactive)
        self.xdegree, self.ydegree = [None]*2
        self.xstructure, self.ystructure = [None]*2

    def __repr__(self):
        rep = super().__repr__()
        return (rep + ',\n' +
                f'x: {repr(self.xdegree)},\n'
                f'xtype: {repr(self.xstructure)},\n'
                f'y: {repr(self.ydegree)},\n'
                f'ytype: {repr(self.ystructure)}')

    def nlterms(self, eq, degree, structure):
        """Set active nonlinear terms/monomials to be optimized"""
        if eq in ('state', 'x'):
            self.xdegree = np.asarray(degree)
            self.xstructure = structure
            # all possible terms
            self.xpowers = combinations(self.n+self.m, degree)
            self.n_nx = self.xpowers.shape[0]
            self.xactive = \
                select_active(self.xstructure, self.n,
                              self.m, self.n, self.xdegree)
            if self.E.size == 0:
                self.E = np.zeros((self.n, self.n_nx))
            # Compute the derivatives of the polynomials zeta and e
            self.xd_powers, self.xd_coeff = poly_deriv(self.xpowers)
        elif eq in ('output', 'y'):
            self.ydegree = np.asarray(degree)
            self.ystructure = structure
            self.ypowers = combinations(self.n+self.m, degree)
            self.n_ny = self.ypowers.shape[0]
            self.yactive = \
                select_active(self.ystructure, self.n,
                              self.m, self.p, self.ydegree)
            if self.F.size == 0:
                self.F = np.zeros((self.p, self.n_ny))
            self.yd_powers, self.yd_coeff = poly_deriv(self.ypowers)

    def output(self, u, t=None, x0=None):
        return dnlsim(self, u, t=t, x0=x0)

    def jacobian(self, x0, weight=False):
        return jacobian(x0, self, weight=weight)


# https://github.com/scipy/scipy/blob/master/scipy/signal/ltisys.py
def dnlsim(system, u, t=None, x0=None):
    """Simulate output of a discrete-time nonlinear system.

    Calculate the output and the states of a nonlinear state-space model.
        x(t+1) = A x(t) + B u(t) + E zeta(x(t),u(t))
        y(t)   = C x(t) + D u(t) + F eta(x(t),u(t))
    where zeta and eta are polynomials whose exponents are given in xpowers and
    ypowers, respectively. The maximum degree in one variable (a state or an
    input) in zeta or eta is given in max_nx and max_ny, respectively. The
    initial state is given in x0.

    """
    # if not isinstance(system, PNLSS):
    #     raise ValueError(f'System must be a PNLSS object {type(system)}')
    #     pass
    # else:
    #     system = NonlinearStateSpace(*system)

    u = np.asarray(u)

    if u.ndim == 1:
        u = np.atleast_2d(u).T

    if t is None:
        out_samples = len(u)
        stoptime = (out_samples - 1) * system.dt
    else:
        stoptime = t[-1]
        out_samples = int(np.floor(stoptime / system.dt)) + 1

    # Pre-build output arrays
    xout = np.empty((out_samples, system.A.shape[0]))
    yout = np.empty((out_samples, system.C.shape[0]))
    tout = np.linspace(0.0, stoptime, num=out_samples)

    # Check initial condition
    if x0 is None:
        xout[0, :] = np.zeros((system.A.shape[1],))
    else:
        xout[0, :] = np.asarray(x0)

    # Pre-interpolate inputs into the desired time steps
    if t is None:
        u_dt = u
    else:
        if len(u.shape) == 1:
            u = u[:, np.newaxis]

        u_dt_interp = interp1d(t, u.transpose(), copy=False, bounds_error=True)
        u_dt = u_dt_interp(tout).transpose()

    # prepare nonlinear part
    repmat_x = np.ones(system.xpowers.shape[0])
    repmat_y = np.ones(system.ypowers.shape[0])
    # Simulate the system
    for i in range(0, out_samples - 1):
        # State equation x(t+1) = A*x(t) + B*u(t) + E*zeta(x(t),u(t))
        zeta_t = np.prod(np.outer(repmat_x, np.hstack((xout[i], u_dt[i])))
                         ** system.xpowers, axis=1)
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]) +
                        np.dot(system.E, zeta_t))
        # Output equation y(t) = C*x(t) + D*u(t) + F*eta(x(t),u(t))
        eta_t = np.prod(np.outer(repmat_y, np.hstack((xout[i], u_dt[i])))
                        ** system.ypowers, axis=1)
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]) +
                      np.dot(system.F, eta_t))

    # Last point
    eta_t = np.prod(np.outer(repmat_y, np.hstack((xout[-1], u_dt[-1])))
                    ** system.ypowers, axis=1)
    yout[-1, :] = (np.dot(system.C, xout[-1, :]) +
                   np.dot(system.D, u_dt[-1, :]) +
                   np.dot(system.F, eta_t))

    return tout, yout, xout


def element_jacobian(samples, A_Edwdx, C_Fdwdx, active):
    """Compute Jacobian of the output y wrt. A, B, and E

    The Jacobian is calculated by filtering an alternative state-space model

    ∂x∂Aᵢⱼ(t+1) = Iᵢⱼx(t) + (A + E*∂ζ∂x)*∂x∂Aᵢⱼ(t)
    ∂y∂Aᵢⱼ(t) = (C + F*∂η∂x)*∂x∂Aᵢⱼ(t)

    where JA = ∂y∂Aᵢⱼ

    Parameters
    ----------
    samples : ndarray
       x, u or zeta corresponding to JA, JB, or JE
    A_Edwdx : ndarray (n,n,NT)
       The result of ``A + E*∂ζ∂x``
    C_Fdwdx : ndarray (p,n,NT)
       The result of ``C + F*∂η∂x``
    active : ndarray
       Array with index of active elements. For JA: np.arange(n**2), JB: n*m or
       JE: xactive

    Returns
    -------
    JA, JB or JE depending on the samples given as input

    See fJNL

    """
    p, n, NT = C_Fdwdx.shape  # Number of outputs and number of states
    # Number of samples and number of inputs in alternative state-space model
    N, npar = samples.shape
    nactive = len(active)  # Number of active parameters in A, B, or E

    out = np.zeros((p, N, nactive))
    for k, activ in enumerate(active):
        # Which column in A, B, or E matrix
        j = np.mod(activ, npar)
        # Which row in A, B, or E matrix
        i = (activ-j)//npar
        # partial derivative of x(0) wrt. A(i,j), B(i,j), or E(i,j)
        Jprev = np.zeros(n)
        for t in range(1, N):
            # Calculate state update alternative state-space model at time t
            # Terms in alternative states at time t-1
            J = A_Edwdx[:, :, t-1] @ Jprev
            # Term in alternative input at time t-1
            J[i] += samples[t-1, j]
            # Calculate output alternative state-space model at time t
            out[:, t, k] = C_Fdwdx[:, :, t] @ J
            # Update previous state alternative state-space model
            Jprev = J

    return out


def jacobian(x0, system, weight=False):
    """Compute the Jacobians of a steady state nonlinear state-space model

    Jacobians of a nonlinear state-space model

        x(t+1) = A x(t) + B u(t) + E zeta(x(t),u(t))
        y(t)   = C x(t) + D u(t) + F eta(x(t),u(t))

    i.e. the partial derivatives of the modeled output w.r.t. the active
    elements in the A, B, E, F, D, and C matrices, fx: JA = ∂y/∂Aᵢⱼ

    x0 : ndarray
        flattened array of state space matrices

    """
    n, m, p = system.n, system.m, system.p
    R, p, npp = system.signal.R, system.signal.p, system.signal.npp
    nfd = npp//2
    # total number of points
    N = R*npp  # system.signal.um.shape[0]
    without_T2 = system.without_T2

    A, B, C, D, E, F = system.extract(x0)

    # Collect states and outputs with prepended transient sample
    x_trans = system.x_mod[system.idx_trans]
    u_trans = system.signal.um[system.idx_trans]
    contrib = np.hstack((x_trans, u_trans)).T
    n_trans = u_trans.shape[0]  # NT

    # E∂ₓζ + A(n,n,NT)
    if E.size == 0:
        A_EdwxIdx = np.zeros(shape=(*A.shape, n_trans))
    else:
        A_EdwxIdx = multEdwdx(contrib, system.xd_powers, np.squeeze(system.xd_coeff),
                              E, n)
    A_EdwxIdx += A[..., None]
    zeta = nl_terms(contrib, system.xpowers).T  # (NT,n_nx)

    # F∂ₓη  (p,n,NT)
    if F.size == 0:
        FdwyIdx = np.zeros(shape=(*C.shape, n_trans))
    else:
        FdwyIdx = multEdwdx(contrib, system.yd_powers, np.squeeze(system.yd_coeff),
                            F, n)
    # Add C to F∂ₓη for all samples at once
    FdwyIdx += C[..., None]
    eta = nl_terms(contrib, system.ypowers).T  # (NT,n_ny)

    # calculate jacobians wrt state space matrices
    JC = np.kron(np.eye(p), system.x_mod)  # (p*N,p*n)
    JD = np.kron(np.eye(p), system.signal.um)  # (p*N, p*m)
    if system.yactive.size:
        JF = np.kron(np.eye(p), eta)  # Jacobian wrt all elements in F
        JF = JF[:, system.yactive]  # all active elements in F. (p*NT,nactiveF)
        JF = JF[system.idx_remtrans]  # (p*N,nactiveF)
    else:
        JF = np.array([]).reshape(p*N, 0)

    # calculate Jacobian by filtering an alternative state-space model
    JA = element_jacobian(x_trans, A_EdwxIdx, FdwyIdx, np.arange(n**2))
    JA = JA.transpose((1, 0, 2)).reshape((p*n_trans, n**2))
    JA = JA[system.idx_remtrans]  # (p*N,n**2)

    JB = element_jacobian(u_trans, A_EdwxIdx, FdwyIdx, np.arange(n*m))
    JB = JB.transpose((1, 0, 2)).reshape((p*n_trans, n*m))
    JB = JB[system.idx_remtrans]  # (p*N,n*m)

    if system.xactive.size:
        JE = element_jacobian(zeta, A_EdwxIdx, FdwyIdx, system.xactive)
        JE = JE.transpose((1, 0, 2)).reshape((p*n_trans, len(system.xactive)))
        JE = JE[system.idx_remtrans]  # (p*N,nactiveE)
    else:
        JE = np.array([]).reshape(p*N, 0)

    jac = np.hstack((JA, JB, JC, JD, JE, JF))[without_T2]
    npar = jac.shape[1]

    # add frequency weighting
    if weight is not False and system.freq_weight:
        # (p*ns, npar) -> (Npp,R,p,npar) -> (Npp,p,R,npar) -> (Npp,p,R*npar)
        jac = jac.reshape((npp, R, p, npar),
                          order='F').swapaxes(1, 2).reshape((-1, p, R*npar),
                                                            order='F')
        # select only the positive half of the spectrum
        jac = fft(jac, axis=0)[:nfd]
        jac = mmul_weight(jac, weight)
        # (nfd,p,R*npar) -> (nfd,p,R,npar) -> (nfd,R,p,npar) -> (nfd*R*p,npar)
        jac = jac.reshape((-1, p, R, npar),
                          order='F').swapaxes(1, 2).reshape((-1, npar), order='F')

        J = np.empty((2*nfd*R*p, npar))
        J[:nfd*R*p] = jac.real
        J[nfd*R*p:] = jac.imag
    elif weight is not False:
        raise ValueError('Time weighting not possible')
    else:
        return jac

    return J
