#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy

import numpy as np
from numpy.fft import fft
from scipy.optimize import least_squares
from scipy.signal.lti_conversion import abcd_normalize
from scipy.signal.ltisys import dlsim

from pyvib.common import lm, mmul_weight, weightfcn

from .lti_conversion import discrete2cont, ss2phys
from .modal import modal_ac


def _atleast_2d_or_none(arg):
    if arg is not None:
        return np.atleast_2d(arg)


class StateSpace():
    def __init__(self, *system, **kwargs):
        """Initialize the state space lti/dlti system."""

        self.inputs = None
        self.outputs = None
        self._dt = None
        self.T1, self.T2 = [None]*2
        self.n, self.m, self.p = [0]*3

        sys = system
        dt = kwargs.pop('dt', True)
        super().__init__(**kwargs)
        self._A, self._B, self._C, self._D = [None]*4
        self.Ac, self.Bc = [None]*2
        self.dt = dt
        if len(system) == 1:  # TODO fix and isinstance(system[0], StateSpace):
            sys = system[0]
            if isinstance(sys, StateSpace):
                sys = sys.A, sys.B, sys.C, sys.D

        if len(sys) == 4:
            self.A, self.B, self.C, self.D = abcd_normalize(*sys)
        else:
            pass
            #raise ValueError(f'Wrong initialization of SS {type(system)}')

    def __repr__(self):
        """Return representation of the `StateSpace` system."""
        return (f'{self.__class__.__name__},\n'
                f'{repr(self.A)},\n'
                f'{repr(self.B)},\n'
                f'{repr(self.C)},\n'
                f'{repr(self.D)},\n'
                f'dt: {repr(self.dt)}')

    @property
    def A(self):
        """State matrix of the `StateSpace` system."""
        return self._A

    @A.setter
    def A(self, A):
        self._A = _atleast_2d_or_none(A)
        self.n = self.A.shape[0]

    @property
    def B(self):
        """Input matrix of the `StateSpace` system."""
        return self._B

    @B.setter
    def B(self, B):
        self._B = _atleast_2d_or_none(B)
        self.m = self.inputs = self.B.shape[-1]

    @property
    def C(self):
        """Output matrix of the `StateSpace` system."""
        return self._C

    @C.setter
    def C(self, C):
        self._C = _atleast_2d_or_none(C)
        self.p = self.outputs = self.C.shape[0]

    @property
    def D(self):
        """Feedthrough matrix of the `StateSpace` system."""
        return self._D

    @D.setter
    def D(self, D):
        self._D = _atleast_2d_or_none(D)

    @property
    def npar(self):
        n, m, p = self.n, self.m, self.p
        return n**2 + n*m + p*n + p*m

    @property
    def dt(self):
        """Return the sampling time of the system."""
        return self._dt

    @dt.setter
    def dt(self, dt):
        self._dt = dt

    def _copy(self, *system):
        """
        Copy the parameters of another `StateSpace` system.
        Parameters
        ----------
        system : instance of `StateSpace`
            The state-space system that is to be copied
        """
        if len(system) == 1 and isinstance(system[0], StateSpace):
            A, B, C, D, dt = (system.A, system.B, system.C, system.D, system.dt)
        elif len(system) == 4:
            A, B, C, D = system
            dt = self.dt
        else:
            raise ValueError('Cannot copy the given system')
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.dt = dt

    def _get_shape(self):
        # n, m, p
        return self.A.shape[0], self.B.shape[1], self.C.shape[0]

    def _get_system(self):
        return (self.A, self.B, self.C, self.D, self.dt)

    def extract(self, x0):
        n, m, p = self.n, self.m, self.p
        A = x0.flat[:n**2].reshape((n,n))
        B = x0.flat[n**2 + np.r_[:n*m]].reshape((n,m))
        C = x0.flat[n**2+n*m + np.r_[:p*n]].reshape((p,n))
        D = x0.flat[n*(p+m+n):].reshape((p,m))
        return A, B, C, D

    def flatten(self):
        """Returns the state space as flattened array"""
        n, m, p = self.n, self.m, self.p
        npar = n**2 + n*m + p*n + p*m

        x0 = np.empty(npar)
        x0[:n**2] = self.A.ravel()
        x0[n**2 + np.r_[:n*m]] = self.B.ravel()
        x0[n**2 + n*m + np.r_[:n*p]] = self.C.ravel()
        x0[n**2 + n*m + n*p:] = self.D.ravel()
        return x0

    def transient(self, T1=None, T2=None):
        """Transient handling. t1: periodic, t2: aperiodic
        Get transient index. Only needed to run once
        """
        self.T1 = T1
        self.T2 = T2
        sig = self.signal
        ns = sig.R * sig.npp
        if T1 is not None:
            # Extract the transient part of the input
            self.idx_trans = transient_indices_periodic(T1, ns)
            self.idx_remtrans = remove_transient_indices_periodic(T1, ns,
                                                                  self.p)
        else:
            self.idx_trans = np.s_[:ns]
            self.idx_remtrans = np.s_[:ns]

        if T2 is not None:
            self.without_T2, NT = remove_transient_indices_nonperiodic(T2,ns,self.p)
        else:
            self.without_T2 = np.s_[:ns]

    def output(self, u, t=None, x0=None):
        system = self._get_system()
        return dlsim(system, u, t=t, x0=x0)

    def simulate(self, u, t=None, x0=None, T1=None, T2=None):
        """
        Return the response of the discrete-time system to input `u` with
        transient handling.

        See :func:`scipy.signal.dlsim` for details.
        """

        # Number of samples
        u = np.atleast_1d(u)
        if u.ndim == 1:
            u = np.atleast_2d(u).T
        ns = u.shape[0]
        if T1 is None:
            T1 = self.T1
            T2 = self.T2
            if T1 is not None:
                idx = self.idx_trans
        else:
            idx = transient_indices_periodic(T1, ns)

        if T1 is not None:
            # Prepend transient samples to the input
            u = u[idx]
        t, y, x = self.output(u, t=t, x0=x0)

        if T1 is not None:
            # remove transient samples. p=1 is correct. TODO why?
            idx = remove_transient_indices_periodic(T1, ns, p=1)
            x = x[idx]
            y = y[idx]
            t = t[idx]

        # save output
        self.x_mod = x
        self.y_mod = y
        return t, y, x

    def to_cont(self, method='zoh', alpha=None):
        """convert to cont. time. Only A and B changes"""
        self.Ac, self.Bc, *_ = \
            discrete2cont(self.A, self.B, self.C, self.D, self.dt,
                          method=method, alpha=alpha)

    @property
    def modal(self, update=False):
        """Calculate modal properties using cont. time matrices"""
        if self.Ac is None or update is True:
            self.to_cont()
        return modal_ac(self.Ac, self.C)

    def to_phys(self, update=False):
        """Calculate state space matrices in physical domain using a similarity
        transform T
        """
        # returns A, B, C, T. T is similarity transform
        if self.Ac is None or update is True:
            self.to_cont()
        return ss2phys(self.Ac, self.Bc, self.C)

class NonlinearStateSpace(StateSpace):
    def __init__(self, *system, **kwargs):
        """Initialize the state space lti/dlti system."""

        sys = system
        E = np.array([])
        F = np.array([])
        if len(system) == 6:
            E, F = system[4:6]
            sys = system[0:4]
        super().__init__(*sys,**kwargs)
        self.E, self.F = E, F

    def __repr__(self):
        rep = super().__repr__()
        idt = rep.rfind('dt')
        inl = rep.find('\n')+1

        return (f'{self.__class__.__name__},\n' +
                rep[inl:idt] +
                f'{repr(self.E)},\n'
                f'{repr(self.F)},\n'
                f'dt: {repr(self.dt)}')

    @property
    def E(self):
        """State matrix of the `StateSpace` system."""
        return self._E

    @E.setter
    def E(self, E):
        self._E = _atleast_2d_or_none(E)

    @property
    def F(self):
        """Input matrix of the `StateSpace` system."""
        return self._F

    @F.setter
    def F(self, F):
        self._F = _atleast_2d_or_none(F)

    @property
    def npar(self):
        xact = self.xactive
        yact = self.yactive
        ne = len(xact)
        nf = len(yact)
        n, m, p = self.n, self.m, self.p
        return n**2 + n*m + p*n + p*m + ne + nf

    def _get_system(self):
        return (self.A, self.B, self.C, self.D, self.E, self.F, self.dt)

    def _copy(self, *system):
        if len(system) == 1 and isinstance(system[0], NonlinearStateSpace):
            A, B, C, D, E, F, dt = (system.A, system.B, system.C, system.D,
                                    system.E, system.F, system.dt)
        elif len(system) == 6:
            A, B, C, D, E, F = system
            dt = self.dt
        else:
            raise ValueError(f'Cannot copy the given system {type(system)}')
        self.A, self.B, self.C, self.D, self.E, self.F, self.dt = \
            A, B, C, D, E, F, dt

    def to_cont(self, method='zoh', alpha=None):
        """convert to cont. time. Only A, B and E changes"""
        Bext = np.hstack((self.B, self.E))
        Dext = np.hstack((self.D, self.F))
        self.Ac, Bcext, *_ = \
            discrete2cont(self.A, Bext, self.C, Dext, self.dt,
                          method=method, alpha=alpha)
        self.Bc = Bcext[:,:self.m]
        self.Ec = Bcext[:,self.m:]

    @property
    def weight(self):
        if self._weight is None:
            self._weight = weightfcn(self.signal.covY)
        return self._weight

    def costfcn(self, x0=None, weight=False):
        if weight is True:
            weight = self.weight
        if x0 is None:
            x0 = self.flatten()
        return costfcn_time(x0, self, weight=weight)

    def extract(self, x0):
        """Extract state space from from flattened array"""
        n, m, p = self.n, self.m, self.p
        # index of active elements
        xact = self.xactive
        yact = self.yactive
        ne = len(xact)
        nf = len(yact)

        E = self.E
        F = self.F
        A = x0.flat[:n**2].reshape((n,n))
        B = x0.flat[n**2 + np.r_[:n*m]].reshape((n,m))
        C = x0.flat[n**2+n*m + np.r_[:p*n]].reshape((p,n))
        D = x0.flat[n*(p+m+n) + np.r_[:p*m]].reshape((p,m))
        E.flat[xact] = x0.flat[n*(p+m+n)+p*m + np.r_[:ne]]
        F.flat[yact] = x0.flat[n*(p+m+n)+p*m+ne + np.r_[:nf]]
        return A, B, C, D, E, F

    def flatten(self):
        """Returns the state space as flattened array"""
        xact = self.xactive
        yact = self.yactive
        ne = len(xact)
        nf = len(yact)
        n, m, p = self.n, self.m, self.p
        npar = n**2 + n*m + p*n + p*m + ne + nf

        x0 = np.empty(npar)
        x0[:n**2] = self.A.ravel()
        x0[n**2 + np.r_[:n*m]] = self.B.ravel()
        x0[n**2 + n*m + np.r_[:n*p]] = self.C.ravel()
        x0[n*(p+m+n) + np.r_[:p*m]] = self.D.ravel()
        x0[n*(p+m+n)+p*m + np.r_[:ne]] = self.E.flat[xact]
        x0[n*(p+m+n)+p*m+ne + np.r_[:nf]] = self.F.flat[yact]
        return x0

class StateSpaceIdent():
    def __init__(self):
        self._weight = None

    def cost(self, x0=None, weight=False):
        if weight is True:
            weight = self.weight
        if x0 is None:
            x0 = self.flatten()
        err = self.costfcn(x0, weight=weight)
        # TODO maybe divide by 2 to match scipy's implementation of minpack
        return np.dot(err, err)

    def optimize(self, method=None, weight=True, info=2, nmax=50, lamb=None,
                 ftol=1e-12, xtol=1e-12, gtol=1e-12, copy=False):
        """Optimize the estimated the nonlinear state space matrices"""
        if weight is True:
            weight = self.weight

        self.freq_weight = True
        if weight is False:
            self.freq_weight = False

        if info:
            print(f'\nStarting {self.__class__.__name__} optimization')

        x0 = self.flatten()
        kwargs = {'weight':weight}
        if method is None:
            res = lm(fun=self.costfcn, x0=x0, jac=self.jacobian, info=info,
                     nmax=nmax, lamb=lamb, ftol=ftol, xtol=xtol, gtol=gtol,
                     kwargs=kwargs)
        else:
            res = least_squares(self.costfcn, x0, self.jacobian, method='lm',
                                x_scale='jac', kwargs=kwargs)

        if copy:
            # restore state space matrices to original
            self._copy(*self.extract(x0))
            nmodel = deepcopy(self)
            nmodel._copy(*self.extract(res['x']))
            nmodel.res = res
            return nmodel

        # update the model with the optimized SS matrices
        self._copy(*self.extract(res['x']))
        self.res = res

    def extract_model(self, y, u, t=None, x0=None, T1=None, T2=None,
                      info=2, copy=False):
        """extract the best model using validation data"""

        models = self.res['x_mat']
        nmodels = models.shape[0]
        ss0 = self.flatten()
        err_rms = np.empty(nmodels)
        if info:
            print(f"{'model':5} | {'rms':12} |")
        for i, ss in enumerate(models):
            self._copy(*self.extract(ss))
            tout, yout, xout = self.simulate(u, t=t, x0=x0, T1=T1, T2=T2)
            err_rms[i] = np.sqrt(np.mean((y - yout)**2))
            if info:
                print(f"{i:5d} | {err_rms[i]:12.8g}")
        # best model on new data set
        i = np.nanargmin(err_rms)
        if info:
            print(f"best model is {i} with RMS {err_rms[i]:12.8g}")
        ss = models[i]
        if copy:
            # restore state space matrices to original
            self._copy(*self.extract(ss0))
            nmodel = deepcopy(self)
            nmodel._copy(*self.extract(ss))
            return nmodel, err_rms

        self._copy(*self.extract(ss))
        return err_rms

def costfcn_time(x0, system, weight=False):
    """Compute the vector of residuals such that the function to mimimize is

    res = ∑ₖ e[k]ᴴ*e[k], where the error is given by
    e = weight*(ŷ - y)
    and the weight is the square inverse of the covariance matrix of `y`
    """

    # TODO fix transient
    # T2 = system.T2
    # p is the actual number of output in the signal, not the system output
    R, p, npp = system.signal.R, system.signal.p, system.signal.npp
    p = system.p
    nfd = npp//2
    # without_T2 = system.without_T2

    # update the state space matrices from x0
    # TODO find a way to avoid explicitly updating the state space model.
    # It is not the expected behavior that calculating the cost should change
    # the model! Right now it is done because simulating is using the systems
    # ss matrices
    system._copy(*system.extract(x0))
    # Compute the (transient-free) modeled output and the corresponding states
    t_mod, y_mod, x_mod = system.simulate(system.signal.um)

    # Compute the (weighted) error signal without transient
    if system.signal._ydm is not None:
        ym = np.hstack((system.signal.ym, system.signal._ydm))
    else:
        ym = system.signal.ym

    err = y_mod - ym  #[without_T2, :p] - system.signal.ym[without_T2]
    if weight is not False and system.freq_weight:
        err = err.reshape((npp,R,p),order='F').swapaxes(1,2)
        # Select only the positive half of the spectrum
        err = fft(err, axis=0)[:nfd]
        err = mmul_weight(err, weight)
        #cost = np.vdot(err, err).real
        err = err.swapaxes(1,2).ravel(order='F')
        err_w = np.hstack((err.real.squeeze(), err.imag.squeeze()))
    elif weight is not False:
        # TODO time domain weighting. Does not work
        err_w = err * weight  # [without_T2]
        #cost = np.dot(err,err)
    else:
        # no weighting
        # TODO are we sure this is the right order?
        return err.ravel(order='F')

    return err_w

def transient_indices_periodic(T1,N):
    """Computes indices for transient handling of periodic signals.

    Computes the indices to be used with a vector u of length N that contains
    (several realizations of) a periodic signal, such that u[indices] has T1[0]
    transient samples prepended to each realization. The starting samples of
    each realization can be specified in T1[1:]. Like this, steady-state data
    can be obtained from a PNLSS model by using u[indices] as an input signal
    to a PNLSS model (see :meth:`pyvib.PNLSS.simulate`) and removing the
    transient samples afterwards (see :func:`remove_transient_indices_periodic`

    Parameters
    ----------
    T1 : int | ndarray(int)
        array that indicates how the transient is handled. The first element
        T1[0] is the number of transient samples that should be prepended to
        each realization. The other elements T1[1:] indicate the starting
        sample of each realization in the signal. If T1 has only one element,
        T1[1] is put to zero, ie. first element.
    N : int
        length of the signal containing all realizations

    Returns
    -------
    indices : ndarray(int)
        indices of a vector u that contains (several realizations of) a
        periodic signal, such that u[indices] has a number of transient samples
        added before each realization

    Examples
    --------
    >>> npp = 1000  # Number of points per period
    >>> R = 2  # Number of phase realizations
    >>> T = 100  # Number of transient samples
    >>> T1 = np.r_[T, np.r_[0:(R-1)*npp+1:npp]]  # Transient handling vector
    >>> N = R*npp  # Total number of samples
    >>> indices = transient_indices_periodic(T1,N)
    indices = np.r_[900:1000, 0:1000, 1900:2000, 1000:2000]
            = [transient samples realization 1, ...
               realization 1, ...
               transient samples realization 2, ...
               realization 2]
    """
    T1 = np.atleast_1d(np.asarray(T1, dtype=int))
    ntrans = T1[0]

    if ntrans != 0:

        if len(T1) == 1:
            # If starting samples of realizations not specified, then we assume
            # the realization start at the first sample
            T1 = np.append(T1, 0)
        # starting index of each realization and length of signal
        T1 = np.append(T1[1:], N)

        indices = np.array([], dtype=int)
        for i in range(len(T1)-1):
            trans = T1[i+1] - 1 - np.mod(np.arange(ntrans)[::-1], T1[i+1]-T1[i])
            normal = np.arange(T1[i],T1[i+1])
            indices = np.hstack((indices, trans, normal))
    else:
        # No transient points => output = all indices of the signal
        indices = np.arange(N)

    return indices

def remove_transient_indices_periodic(T1,N,p):
    """Computes indices for transient handling for periodic signals after
    filtering

    Let u be a vector of length N containing (several realizations of) a
    periodic signal. Let uTot be a vector containing the signal(s) in u with
    T1[0] transient points prepended to each realization (see
    :func:`transient_indices_periodic`). The starting samples of each
    realization can be specified in T1[1:]. Let yTot be a vector/matrix
    containing the p outputs of a PNLSS model after applying the input uTot.
    Then this function computes the indices to be used with the vectorized form
    of yTot such that the transient samples are removed from yTot, i.e. y =
    yTot[indices] contains the steady-state output(s) stacked on top of each
    other.

    Parameters
    ----------
    T1 : ndarray(int)
        vector that indicates how the transient is handled. The first element
        T1[0] is the number of transient samples that were prepended to each
        realization. The other elements T1[1:] indicate the starting sample
        of each realization in the input signal. If T1 has only one element,
        T1[1] is put to zero.
    N : int
        length of the input signal containing all realizations
    p : int
        number of outputs

    Returns
    -------
    indices : ndarray(int)
        If uTot is a vector containing (several realizations of) a periodic
        signal to which T1[0] transient points were added before each
        realization, and if yTot is the corresponding output vector (or matrix
        if more than one output), then indices is such that the transient
        points are removed from y = yTot.flat[indices]. If p > 1, then indices
        is a vector and y = yTot.flat[indices] is a vector with the steady
        state outputs stacked after each other.

    Examples
    --------
    >>> npp = 1000  # Number of points per period
    >>> R = 2  # Number of phase realizations
    >>> T = 100  # Number of transient samples
    >>> T1 = np.r_[T, np.r_[0:(R-1)*npp+1:npp]]  # Transient handling vector
    >>> N = R*npp  # Total number of samples
    >>> indices_tot = transient_indices_periodic(T1,N)
    indices_tot = np.r_[900:1000, 0:1000, 1900:2000, 1000:2000]
    >>> p = 1  # One output
    >>> indices_removal = remove_transient_indices_periodic(T1,N,p)
    np.r_[100:1100, 1200:2200]
    >>> indices_tot[indices_removal]
    np.r_[:2000]  # [realization 1, realization 2]
    >>> p = 2  # More than one output
    >>> indices_removal = remove_transient_indices_periodic(T1,N,p)
    np.r_[100:1100, 1200:2200, 2300:3300, 3400:4400]

    Let u be a vector containing `[input realization 1, input realization 2]`
    then `uTot = u[indices_tot]` is a vector containing::

        [transient samples realization 1, input realization 1,
         transient samples realization 2, input realization 2]

    Let y1 be a vector containing the first output and y2 be a vector
    containing the second output when applying uTot as an input to a
    PNLSS model, and let `yTot = [y1, y2].T` be a 2 x 2200 matrix with y1
    and y2 in its first and second row, respectively.
    Note that `y1 = yTot.flat[:2200]` and `y2 = yTot.flat[2200:4400]`
    Then `yTot.flat[indices_removal] = np.r_[y1[100:1100], y1[1200:2200],
                                             y2[100:1100], y2[1200:2200]]`::

        [output 1 corresponding to input realization 1,
         output 1 corresponding to input realization 2,
         output 2 corresponding to input realization 1,
         output 2 corresponding to input realization 2]

    """
    T1 = np.atleast_1d(np.asarray(T1, dtype=int))
    ntrans = T1[0]

    if ntrans == 0:
        return np.arange(N)

    if len(T1) == 1:
        # If starting samples of realizations not specified, then we assume
        # the realization start at the first sample
        T1 = np.append(T1, 0)

    # starting index of each realization and length of signal
    T1 = np.append(T1[1:], N)

    indices = np.array([], dtype=int)
    for i in range(len(T1)-1):
        # Concatenate indices without transient samples
        indices = np.hstack((indices,
                             np.r_[T1[i]:T1[i+1]] + (i+1)*ntrans))

    # TODO This is not correct for p>1. We still store y.shape -> (N,p)
    # UPDATE 25/02: maybe correct. Gives correct output, see examples
    if p > 1:
        # Total number of samples per output = number of samples without + with
        # transients
        nt = N + ntrans*(len(T1)-1)

        tmp = np.empty(p*N, dtype=int)
        for i in range(p):
            # Stack indices without transient samples on top of each other
            tmp[i*N:(i+1)*N] = indices + i*nt
        indices = tmp

    return indices

def remove_transient_indices_nonperiodic(T2,N,p):
    """Remove transients from arbitrary data.

    Computes the indices to be used with a (N,p) matrix containing p output
    signals of length N, such that y[indices] contains the transient-free
    output(s) of length NT stacked on top of each other (if more than one
    output). The transient samples to be removed are specified in T2 (T2 =
    np.arange(T2) if T2 is scalar).

    Parameters
    ----------
    T2 : int
        scalar indicating how many samples from the start are removed or array
        indicating which samples are removed
    N : int
        length of the total signal
    p : int
        number of outputs

    Returns
    -------
    indices : ndarray(int)
        vector of indices, such that y[indices] contains the output(s) without
        transients. If more than one output (p > 1), then y[indices] stacks the
        transient-free outputs on top of each other.
    nt : int
        length of the signal without transients

    Examples
    --------
    # One output, T2 scalar
    >>> N = 1000 # Total number of samples
    >>> T2 = 200  # First 200 samples should be removed after filtering
    >>> p = 1  # One output
    >>> indices, NT = remove_transient_indices_nonperiodic(T2,N,p)
    np.r_[200:1000]  # Indices of the transient-free output
    NT = 800  # Number of samples in the transient-free output

    # Two outputs, T2 scalar
    >>> N = 1000  # Total number of samples
    >>> T2 = 200  # First 200 samples should be removed after filtering
    >>> p = 2  # Two outputs
    >>> indices, NT = remove_transient_indices_nonperiodic(T2,N,p)
    np.r_[200:1000, 1200:2000]
    NT = 800
    If y = [y1, y2] is a 1000 x 2 matrix with the two outputs y1 and y2, then
    y[indices] = [y1(200:1000]
                  y2(200:1000)]
    is a vector with the transient-free outputs stacked on top of each other

    One output, T2 is a vector
    >>> N1 = 1000  # Number of samples in a first data set
    >>> N2 = 500  # Number of samples in a second data set
    >>> N = N1 + N2  # Total number of samples
    >>> T2_1 = np.r_[:200]  # Transient samples in first data set
    >>> T2_2 = np.r_[:100]  # Transient samples in second data set
    >>> T2 = np.r_[T2_1, N1+T2_2]  # Transient samples
    >>> p = 1  # One output
    >>> indices, NT = remove_transient_indices_nonperiodic(T2,N,p)
    np.r_[200:1000, 1100:1500]
    NT = 1200
    """

    if T2 is None:
        return np.s_[:N], N

    if isinstance(T2, (int, np.integer)):  # np.isscalar(T2):
        # Remove all samples up to T2
        T2 = np.arange(T2)

    T2 = np.atleast_1d(np.asarray(T2, dtype=int))
    # Remove transient samples from the total
    without_T2 = np.delete(np.arange(N), T2)

    # Length of the transient-free signal(s)
    NT = len(without_T2)
    if p > 1:  # for multiple outputs
        indices = np.zeros(p*NT, dtype=int)
        for i in range(p):
            # Stack indices for each output on top of each other
            indices[i*NT:(i+1)*NT] = without_T2 + i*N
    else:
        indices = without_T2

    return indices, NT
