#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft
from scipy.linalg import norm, solve

# from .helper.modal_plotting import plot_frf, plot_stab
from .lti_conversion import discrete2cont
from .nlss import NLSS
from .statespace import NonlinearStateSpace, StateSpaceOptimization
from .subspace import subspace


class FNSI(NLSS, NonlinearStateSpace, StateSpaceOptimization):
    """Identify nonlinear subspace model in frequency domain

    x(t+1) = A x(t) + B u(t) + E g(y(t),u(t))
    y(t)   = C x(t) + D u(t)

    Nonlinear forces are concatenated with the input, forming the extended
    input `e = [u, -g(y,ẏ)].T`. Thus the user needs to know the form of `g`;
    ex. qubic polynomial, tanh(ẏ), etc., and which DOF the nonlinearity is
    connected to. The latter is specified with eg. `w=[0,1]` for y = [y,ẏ].

    This method can estimate the coefficients of `g`; either in physical space
    as a frequency dependent variable, or in discrete form as the the
    coefficients of `E`.

    The difference between :class:`~nlss.NLSS` and this, is that `NLSS` is a
    black-box model with nonlinearities in both state- and output equation,
    where this is a grey-box model with only (user-specified)
    nonlinearities in the state equation. That requires the nonlinearity to be:
    - localized (fx. distributed geometric NL cannot be identified with FNSI)
    - static
    -

    Example
    -------
    >>> from pyvib.nonlinear_elemets import Tanhdryfriction
    >>> nlx = [Tanhdryfriction(eps=0.01, w=[0,1]])
    >>> fnsi = FNSI()
    >>> fnsi.set_signal(sig)
    >>> fnsi.add_nl(nlx=nlx)
    >>> fnsi.estimate(n=2, r=5, weight=weight)
    >>> fnsi.transient(T1)
    >>> fnsi.optimize(lamb=100, weight=weight, nmax=25)

    Notes
    -----
    "Grey-box state-space identification of nonlinear mechanical vibrations"
    https://sci-hub.tw/10.1080/00207179.2017.1308557
    FNSI method by J.P. Noël
    "Frequency-domain subspace identification for nonlinear mechanical
    systems"
    https://sci-hub.tw/j.ymssp.2013.06.034
    """

    def __init__(self, *system, **kwargs):
        if len(system) == 1:  # and isinstance(system[0], StateSpace):
            sys = system
            kwargs['dt'] = sys[0].dt
        else:  # given as A,B,C,D
            sys = system
            kwargs['dt'] = 1  # unit sampling

        super().__init__(*sys, **kwargs)
        self.r = None
        self.Ff = np.array([])

    def to_cont(self, method='zoh', alpha=None):
        """Convert to discrete time. Only A and B changes for zoh method"""
        Bext = np.hstack((self.B, -self.E))
        Dext = np.hstack((self.D, -self.Ff))
        Ac, Bcext, Cc, Dcext = \
            discrete2cont(self.A, Bext, self.C, Dext, self.dt, method, alpha)

        Bc = Bcext[:, :self.m]
        Ec = Bcext[:, self.m:]
        Dc = Dcext[:, :self.m]
        Fc = Dcext[:, self.m:]
        return Ac, Bc, Cc, Dc, Ec, Fc

    def ext_input(self, fmin=None, fmax=None):
        """Form the extended input and output

        The concatenated extended input vector `e=[u, g(y,ẏ)].T`

        Returns
        -------
        E : ndarray(npp,m+nnl) (complex)
            FFT of the concatenated extended input vector e = [u, -g].T
        Y : ndarray(npp,p) (complex)
            FFT of y.
        """
        sig = self.signal
        self.p, self.m = sig.p, sig.m
        npp = sig.npp
        assert sig.R == 1, 'For FNSI there can only be one realization in data'

        fs = 1/self.dt
        if fmin is not None and fmax is not None:
            f1 = int(np.floor(fmin/fs * npp)) + 1
            f2 = int(np.ceil(fmax/fs * npp))
            self.lines = np.arange(f1, f2+1)
        else:
            self.lines = sig.lines

        # if the data is not truly periodic, there is a slight difference
        # between doing Y=fft(sig.y); Ymean = np.sum(Y) / sig.P and taking the
        # fft directly of the averaged time signal as here.
        Umean = fft(sig.um, axis=0)
        Ymean = fft(sig.ym, axis=0)

        # In case of no nonlinearities
        if self.nlx.n_nl == 0:
            scaling = []
            E = Umean
        else:
            # only output-based NLs
            fnl = self.nlx.fnl(0, sig.ym, 0).T
            nnl = fnl.shape[1]

            scaling = np.zeros(nnl)
            for j in range(nnl):
                scaling[j] = np.std(sig.u[:, 0]) / np.std(fnl[:, j])
                fnl[:, j] *= scaling[j]

            FNL = fft(fnl, axis=0)
            # concatenate to form extended input spectra matrix
            E = np.hstack((Umean, -FNL))

        U = E[self.lines]/np.sqrt(npp)
        Y = Ymean[self.lines]/np.sqrt(npp)
        scaling = scaling
        return U, Y, scaling

    def estimate(self, n, r, bd_method='opt', fmin=None, fmax=None, weight=None):
        self.r = r
        self.n = n
        # set active elements now the system size is specified
        self._set_active()

        # form the extended input
        U, Y, scaling = self.ext_input(fmin=fmin, fmax=fmax)

        # normalized frequency [0-0.5]
        freq = self.lines / self.signal.npp
        covG = False
        G = None
        Ad, Bd, Cd, Dd, z, isstable = \
            subspace(G, covG, freq, self.n, self.r, U, Y, bd_method)

        # extract nonlinear coefficients
        n_nx = self.nlx.n_nl
        E = np.zeros((n, n_nx))
        F = np.zeros((self.p, n_nx))
        for i in range(n_nx):
            E[:, i] = - scaling[i]*Bd[:, self.m+i]
            F[:, i] = - scaling[i]*Dd[:, self.m+i]

        self.A = Ad
        self.B = Bd[:, :self.m]
        self.C = Cd
        self.D = Dd[:, :self.m]
        self.E = E
        self.Ff = F

    def nl_coeff(self, iu):
        """Form the extended FRF (transfer function matrix) He(ω) and extract
        nonlinear coefficients
        G(ω) is the linear FRF matrix, eq. (46)
        He(ω) is formed using eq (47)

        Parameters
        ----------
        iu : int
            Location of exciting force.

        Returns
        -------
        knl : complex ndarray(F, n_nx) or empty
            The nonlinear coefficients (frequency-dependent and complex-valued)
        G(ω) : complex ndarray(F, p)
            Estimate of the linear FRF
        He(ω) : complex ndarray
            The extended FRF (transfer function matrix)
        """
        sig = self.signal
        lines = self.lines
        p, m, nnl = self.p, self.m, self.nlx.n_nl
        Ac, Bc, Cc, Dc, Ec, Fc = self.to_cont(method='zoh')
        # Recombine E and F. They were extracted as negative part of B and D
        Bext = np.hstack((Bc, -Ec))
        Dext = np.hstack((Dc, -Fc))

        freq = np.arange(sig.npp)/sig.npp/self.dt
        F = len(lines)

        # just return in case of no nonlinearities
        if nnl == 0:
            knl = np.empty(shape=(0, 0))
        else:
            knl = np.empty((F, nnl), dtype=complex)

        # Determine which dofs are connected.
        inl1 = np.zeros(nnl, dtype=int)
        inl2 = np.zeros(nnl, dtype=int)
        # TODO. This only works for MAX two connected dofs. Also it does not
        # take the sign(in w) into account. See the - in knl calc.
        for i, nl in enumerate(self.nlx.nls):
            # each nonzero results in a row in idx
            idx = np.argwhere(np.atleast_2d(nl.w))
            inl1[i] = idx[0,1]
            if idx.shape[0] == 1:  # connected to ground
                inl2[i] = -1
            else:
                inl2[i] = idx[1,1]

        # Extra +1 zeros for p in He is for ground connections
        G = np.empty((F, p), dtype=complex)
        He = np.empty((F, p+1, m+nnl), dtype=complex)
        He[:, -1, :] = 0

        In = np.eye(*Ac.shape, dtype=complex)
        for k in range(F):
            # eq. 47
            He[k, :-1, :] = Cc @ solve(In*2j*np.pi *
                                       freq[lines[k]] - Ac, Bext) + Dext

        for i in range(nnl):
            knl[:, i] = -He[:, iu, m+i] / (He[:, inl1[i], 0]-He[:, inl2[i], 0])

        G = He[:, :p, 0]

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
            # TODO this is only for polynomial nl
            #exponent = 'x'.join(str(x) for x in self.xpowers[i])
            #print('exp: {:s}\t ℝ(mu) {:.4e}\t 𝕀(mu)  {:.4e}'.
            #      format(exponent, *mu_mean))
            print(f' Ratio log₁₀(ℝ(mu)/𝕀(mu))= {ratio:0.2f}')
