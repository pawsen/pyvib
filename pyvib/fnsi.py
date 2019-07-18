#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft
from scipy.interpolate import interp1d
from scipy.linalg import norm, solve

from .common import mmul_weight
from .helper.modal_plotting import plot_frf, plot_stab
from .polynomial import multEdwdx, nl_terms, poly_deriv
from .statespace import NonlinearStateSpace, StateSpaceIdent
from .subspace import modal_list, subspace


class FNSI(NonlinearStateSpace, StateSpaceIdent):
    def __init__(self, signal, *system, **kwargs):
        self.signal = signal
        kwargs['dt'] = 1/signal.fs
        super().__init__(*system, **kwargs)

        # actually these should be reintialized after SS matrices are estimated
        self.xpowers = np.empty(shape=(0,self.m+self.n))
        self.ypowers = np.empty(shape=(0,self.m+self.n))
        self.xactive = np.array([],dtype=int)
        self.yactive = np.array([],dtype=int)
        self.n_nx = len(self.xactive)
        self.n_ny = len(self.yactive)

    def nlterms(self, eq, powers):
        if eq in ('state', 'x'):
            self.xpowers = np.atleast_2d(powers)
            self.xd_powers, self.xd_coeff = poly_deriv(self.xpowers)
            self.n_nx = self.xpowers.shape[0]
            if self.E.size == 0:  # E and F is reinitialized when estimated
                self.E = np.zeros((self.n, self.n_nx))
        elif eq in ('output', 'y'):
            self.ypowers = np.atleast_2d(powers)
            self.yd_powers, self.yd_coeff = poly_deriv(self.ypowers)
            self.n_ny = self.ypowers.shape[0]
            if self.F.size == 0:
                self.F = np.zeros((self.p, self.n_ny))

    def output(self, u, t=None, x0=None):
        return dnlsim(self, u, t=t, x0=x0)

    def jacobian(self, x0, weight=False):
        return jacobian(x0, self, weight=weight)

    def ext_input(self, fmin=None, fmax=None, vel=False):
        """Form the extended input and output

        The concatenated extended input vector e(t), is e=[u(t), g(t)].T, see
        eq (5). Notice that the stacking order is reversed here.
        u(t) is the input force and g(y(t),ẏ(t)) is the functional nonlinear
        force calculated from the specified polynomial nonlinearity, see eq.(2)

        Returns
        -------
        E : ndarray (complex)
            FFT of the concatenated extended input vector e(t)
        Y : ndarray (complex)
            FFT of y.

        Notes
        -----
        Method by J.P Noel. Described in article
        "Frequency-domain subspace identification for nonlinear mechanical
        systems"
        http://dx.doi.org/10.1016/j.ymssp.2013.06.034
        Equation numbers refers to this article

        """
        sig = self.signal
        npp = sig.npp

        fs = 1/self.dt
        if fmin is not None and fmax is not None:
            f1 = int(np.floor(fmin/fs * npp))
            f2 = int(np.ceil(fmax/fs * npp))
            self.flines = np.arange(f1,f2+1)
        else:
            self.flines = sig.lines

        # if the data is not truly periodic, there is a slight difference
        # between doing Y=fft(sig.y); Ymean = np.sum(Y) / sig.P and taking the
        # fft directly of the averaged time signal as here.
        Umean = fft(sig.um,axis=0)
        Ymean = fft(sig.ym,axis=0)
        # If the nonlinear force is formed using the velocity we add it to Yext
        if vel is True:
            Ydmean = fft(sig.ydm,axis=0)

        Yext = np.hstack((Ymean, Ydmean)) if vel is True else Ymean

        # In case of no nonlinearities
        if self.xpowers.size == 0:
            scaling = []
            E = Umean
        else:
            # ynl: [npp, 2*p]
            ynl = np.hstack((sig.ym, sig.ydm)) if vel is True else sig.ym
            nnl = self.n_nx
            repmat_x = np.ones(nnl)

            # einsum does np.outer(repmat_x, ynl[i]) for all i
            fnl = np.prod(np.einsum('i,jk->jik',repmat_x, ynl)**self.xpowers,
                          axis=2)

            scaling = np.zeros(nnl)
            for j in range(nnl):
                scaling[j] = np.std(sig.u[:,0]) / np.std(fnl[:,j])
                fnl[:,j] *= scaling[j]

            FNL = fft(fnl, axis=0)
            # concatenate to form extended input spectra matrix
            E = np.hstack((Umean, -FNL))

        U = E[self.flines]/np.sqrt(npp)
        Y = Yext[self.flines]/np.sqrt(npp)
        scaling = scaling
        return U, Y, scaling

    def estimate(self, n, r, bd_method='explicit', fmin=None, fmax=None,
                 vel=False):
        # form the extended input
        U, Y, scaling = self.ext_input(fmin=fmin, fmax=fmax, vel=vel)

        self.r = r
        self.n = n

        # normalized frequency [0-0.5]
        freq = self.flines / self.signal.npp
        covG = False
        G = None
        Ad, Bd, Cd, Dd, z, isstable = \
            subspace(G, covG, freq, self.n, self.r, U, Y, bd_method)

        # the number of input p, might be different for the data and the model,
        # if velocities are included in the model.
        self.p = Cd.shape[0]
        m, p = self.signal.m, self.p

        # extract nonlinear coefficients
        n_nx = self.n_nx
        E = np.zeros((n, n_nx))
        F = np.zeros((p, n_nx))
        for i in range(n_nx):
            E[:,i] = - scaling[i]*Bd[:,m+i]
            F[:,i] = - scaling[i]*Dd[:,m+i]

        self.A = Ad
        self.B = Bd[:,:m]
        self.C = Cd
        self.D = Dd[:,:m]
        self.E = E
        self.F = F
        if vel is True and self.xpowers.size == 0:
            self.xpowers = np.empty(shape=(0,self.p))
        self.xactive = np.arange(E.size)
        self.yactive = np.arange(0)  # F.size)

    def nl_coeff(self, iu):
        """Form the extended FRF (transfer function matrix) He(ω) and extract
        nonlinear coefficients
        G(ω) is the linear FRF matrix, eq. (46)
        He(ω) is formed using eq (47)
        Parameters
        ----------
        iu : int
            The location of the force.
        Returns
        -------
        knl : ndarray(complex)
            The nonlinear coefficients (frequency-dependent and complex-valued)
        G(ω) : ndarray(complex)
            Estimate of the linear FRF
        He(ω) : ndarray(complex)
            The extended FRF (transfer function matrix)
        """
        sig = self.signal
        flines = self.flines
        p, m, n_nx = self.p, self.m, self.n_nx
        if self.Ac is None:
            self.to_cont()
        Ac = self.Ac
        C = self.C
        # subtract E and F as they are extracted as negative part of B and D
        Bext = np.hstack((self.Bc, -self.Ec))
        Dext = np.hstack((self.D, -self.F))

        freq = np.arange(sig.npp)/self.dt/sig.npp
        F = len(flines)

        nnl = n_nx
        # just return in case of no nonlinearities
        if nnl == 0:
            knl = np.empty(shape=(0,0))
        else:
            knl = np.empty((nnl,F),dtype=complex)

        # Extra rows of zeros in He is for ground connections
        # It is not necessary to set inl's connected to ground equal to l, as
        # -1 already point to the last row.
        G = np.empty((p, F), dtype=complex)
        He = np.empty((p+1, m+nnl, F), dtype=complex)
        He[-1,:,:] = 0

        In = np.eye(*Ac.shape,dtype=complex)
        for k in range(F):
            # eq. 47
            He[:-1,:,k] = C @ solve(In*2j*np.pi*freq[flines[k]] - Ac, Bext) + Dext

            for nl in range(n_nx):
                # number of nonlin connections for the given nl type
                idx = 0
                knl[nl,k] = He[iu, m+nl, k] / (He[idx,0,k] - He[-1,0,k])

            for j, dof in enumerate(range(p)):
                G[j,k] = He[dof, 0, k]

        self.knl = knl
        return G, knl

    @property
    def knl_str(self):
        for i, knl in enumerate(self.knl):
            mu_mean = np.zeros(2)
            mu_mean[0] = np.mean(np.real(knl))
            mu_mean[1] = np.mean(np.imag(knl))
            # ratio of 1, is a factor of 10. 2 is a factor of 100, etc
            ratio = np.log10(np.abs(mu_mean[0]/mu_mean[1]))
            exponent = 'x'.join(str(x) for x in self.xpowers[i])
            print('exp: {:s}\t ℝ(mu) {:.4e}\t 𝕀(mu)  {:.4e}'.
                  format(exponent, *mu_mean))
            print(f' Ratio log₁₀(ℝ(mu)/𝕀(mu))= {ratio:0.2f}')


def dnlsim(system, u, t=None, x0=None):
    """Simulate output of a discrete-time nonlinear system.

    Calculate the output and the states of a nonlinear state-space model.
        x(t+1) = A x(t) + B u(t) + E zeta(y(t),ẏ(t))
        y(t)   = C x(t) + D u(t) + F zeta(y(t),ẏ(t))
    where zeta is a vector of polynomials whose exponents are given in xpowers.
    The initial state is given in x0.

    """
    #if not isinstance(system, FNSI):
    #    raise ValueError(f'System must be a FNSI object {type(system)}')

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
    # Simulate the system
    for i in range(0, out_samples - 1):
        # Output equation y(t) = C*x(t) + D*u(t)
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]))
        # State equation x(t+1) = A*x(t) + B*u(t) + E*zeta(y(t),ẏ(t))
        zeta_t = np.prod(np.outer(repmat_x, yout[i])**system.xpowers,
                         axis=1)
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]) +
                        np.dot(system.E, zeta_t))

    # Last point
    # zeta_t = np.hstack((u_dt[-1, :],xout[-1,idx]**system.xpowers))
    yout[-1, :] = (np.dot(system.C, xout[-1, :]) +
                   np.dot(system.D, u_dt[-1, :]))

    return tout, yout, xout

def jacobian(x0, system, weight=False):
    """Compute the Jacobians of a steady state gray-box state-space model

    Jacobians of a gray-box state-space model

        x(t+1) = A x(t) + B u(t) + E zeta(y(t),ẏ(t))
        y(t)   = C x(t) + D u(t) + F zeta(y(t),ẏ(t))

    i.e. the partial derivatives of the modeled output w.r.t. the active
    elements in the A, B, C, D, E and F matrices, fx: JA = ∂y/∂Aᵢⱼ

    x0 : ndarray
        flattened array of state space matrices
    """
    n, m, p = system.n, system.m, system.p
    R, npp = system.signal.R, system.signal.npp
    nfd = npp//2
    # total number of points
    N = R*npp  # system.signal.um.shape[0]
    # without_T2 = system.without_T2

    A, B, C, D, E, F = system.extract(x0)

    # Collect states and outputs with prepended transient sample
    y_trans = system.y_mod[system.idx_trans]
    x_trans = system.x_mod[system.idx_trans]
    u_trans = system.signal.um[system.idx_trans]
    # (n_var, nt)
    # contrib = np.atleast_2d(np.hstack((y_trans)).T)
    contrib = y_trans.T
    n_trans = u_trans.shape[0]  # NT

    # E∂ₓζ
    ny = contrib.shape[0]
    Edwxdx = multEdwdx(contrib,system.xd_powers,system.xd_coeff, E,ny)
    zeta = nl_terms(contrib, system.xpowers).T  # (NT,n_nx)

    # calculate jacobians wrt state space matrices
    JC = np.kron(np.eye(p), system.x_mod)  # (p*N,p*n)
    JD = np.kron(np.eye(p), system.signal.um)  # (p*N, p*m)

    # calculate Jacobian by filtering an alternative state-space model
    JA = element_jacobian(x_trans, A, C, Edwxdx, None, np.arange(n**2))
    JA = JA.transpose((1,0,2)).reshape((p*n_trans, n**2))
    JA = JA[system.idx_remtrans]  # (p*N,n**2)

    JB = element_jacobian(u_trans, A, C, Edwxdx, None, np.arange(n*m))
    JB = JB.transpose((1,0,2)).reshape((p*n_trans, n*m))
    JB = JB[system.idx_remtrans]  # (p*N,n*m)

    if system.xactive.size:
        JE = element_jacobian(zeta, A, C, Edwxdx, None, system.xactive)
        JE = JE.transpose((1,0,2)).reshape((p*n_trans, len(system.xactive)))
        JE = JE[system.idx_remtrans]  # (p*N,nactiveE)
    else:
        JE = np.array([]).reshape(p*N,0)
    if system.yactive.size:
        JF = np.kron(np.eye(p), zeta)  # Jacobian wrt all elements in F
        JF = JF[:,system.yactive]  # all active elements in F. (p*NT,nactiveF)
        JF = JF[system.idx_remtrans]  # (p*N,nactiveF)
    else:
        JF = np.array([]).reshape(p*N,0)

    jac = np.hstack((JA, JB, JC, JD, JE, JF))  # [without_T2]
    npar = jac.shape[1]

    # add frequency weighting
    if weight is not False and system.freq_weight:
        # (p*ns, npar) -> (Npp,R,p,npar) -> (Npp,p,R,npar) -> (Npp,p,R*npar)
        jac = jac.reshape((npp,R,p,npar),
                          order='F').swapaxes(1,2).reshape((-1,p,R*npar),
                                                           order='F')
        # select only the positive half of the spectrum
        jac = fft(jac, axis=0)[:nfd]
        jac = mmul_weight(jac, weight)
        # (nfd,p,R*npar) -> (nfd,p,R,npar) -> (nfd,R,p,npar) -> (nfd*R*p,npar)
        jac = jac.reshape((-1,p,R,npar),
                          order='F').swapaxes(1,2).reshape((-1,npar), order='F')

        J = np.empty((2*nfd*R*p,npar))
        J[:nfd*R*p] = jac.real
        J[nfd*R*p:] = jac.imag
    elif weight is not False:
        raise ValueError('Time weighting not possible')
    else:
        return jac

    return J

def element_jacobian(samples, A, C, Edwdy, Fdwdy, active):
    """Compute Jacobian of the output y wrt. A, B, and E

    The Jacobian is calculated by filtering an alternative state-space model

    ∂x∂Aᵢⱼ(t+1) = Iᵢⱼx(t) + A*∂x∂Aᵢⱼ(t) + E∂ζ∂y*∂y∂Aᵢⱼ(t)
    ∂y∂Aᵢⱼ(t) = C*∂x∂Aᵢⱼ(t) + F*∂η∂y*∂y∂Aᵢⱼ(t)

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
    # Number of outputs and number of states
    p, n = C.shape
    n_nx, ny, NT = Edwdy.shape
    # Number of samples and number of inputs in alternative state-space model
    N, npar = samples.shape
    nactive = len(active)  # Number of active parameters in A, B, or E

    out = np.zeros((p,N,nactive))
    for k, activ in enumerate(active):
        # Which column in A, B, or E matrix
        j = np.mod(activ, npar)
        # Which row in A, B, or E matrix
        i = (activ-j)//npar
        # partial derivative of x(0) wrt. A(i,j), B(i,j), or E(i,j)
        Jprev = np.zeros(n)
        for t in range(0,N-1):
            # Calculate output alternative state-space model at time t
            out[:,t,k] = C @ Jprev
            # Calculate state update alternative state-space model at time t
            J = A @ Jprev + Edwdy[:,:,t] @ out[:,t,k]
            J[i] += samples[t,j]
            # Update previous state alternative state-space model
            Jprev = J
        out[:,-1,k] = C @ J

    return out
