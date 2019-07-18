#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.fft import fft
from scipy.interpolate import interp1d
from scipy.special import comb

from .common import mmul_weight
from .polynomial import multEdwdx, nl_terms, poly_deriv
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
        self.xpowers = np.empty(shape=(0,self.m+self.n))
        self.ypowers = np.empty(shape=(0,self.m+self.n))
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
                select_active(self.xstructure,self.n,self.m,self.n,self.xdegree)
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
                select_active(self.ystructure,self.n,self.m,self.p,self.ydegree)
            if self.F.size == 0:
                self.F = np.zeros((self.p, self.n_ny))
            self.yd_powers, self.yd_coeff = poly_deriv(self.ypowers)

    def output(self, u, t=None, x0=None):
        return dnlsim(self, u, t=t, x0=x0)

    def jacobian(self, x0, weight=False):
        return jacobian(x0, self, weight=weight)


def combinations(n, degrees):
    """Lists all nonlinear terms in a multivariate polynomial.

    Lists the exponents of all possible monomials in a multivariate polynomial
    with n inputs. Only the nonlinear degrees in ``degrees`` are considered.

    Parameters
    ----------
    n: int
        number of inputs
    degrees: ndarray
        array with the degrees of nonlinearity

    Returns
    -------
    monomials : ndarray(ncomb,n)
        matrix of exponents

    Examples
    --------
    A polynomial with all possible quadratic and cubic terms in the variables x
    and y contains the monomials x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, and y*y*y.
    >>> combinations(2,[2, 3])
    array([[2, 0],  #  -> x^2 * y^0 = x*x
           [1, 1],  #  -> x^1 * y^1 = x*y
           [0, 2],  #  -> x^0 * y^2 = y*y
           [3, 0],  #  -> x^3 * y^0 = x*x*x
           [2, 1],  #  -> x^2 * y^1 = x*x*y
           [1, 2],  #  -> x^1 * y^2 = x*y*y
           [0, 3]]) #  -> x^0 * y^3 = y*y*y

    Element (i,j) of ``out`` indicates the power to which variable j is raised
    in monomial i. For example, out[4] = [2,1], which means that the fifth
    monomial is equal to x^2*y^1 = x*x*y.

    """
    degrees = np.asarray(degrees)
    if not np.issubdtype(degrees.dtype, np.integer):
        raise ValueError('wrong type in degrees. Should only be int. Is {}'.
                         format(degrees.dtype))
    # Consider only nonlinear degrees
    degrees = degrees[degrees > 1]

    # Determine total number of combinations/monomials
    ncomb = 0
    for degree in degrees:
        ncomb += comb(n+degree-1, degree, exact=True)

    # List the exponents of each input in all monomials
    monomials = np.zeros((ncomb,n),dtype=int)
    idx = 0  # Running index indicating the last used row in out
    for degree in degrees:
        # All combinations in homogeneous polynomial of degree
        comb_i = hom_combinations(n,degree)
        ncomb_i = comb_i.shape[0]
        monomials[idx: idx+ncomb_i] = comb_i
        idx += ncomb_i

    return monomials


def hom_combinations(n,degree):
    """Lists the exponents of all possible terms in a homogeneous polynomial
    monomial representation, e.g. [1 2] represents x1*x2**2

    Examples
    --------
    >>> hom_combinations(2,2)
    array([[2, 0],
           [1, 1],
           [0, 2]])
    """

    # Number of combinations in homogeneous polynomial
    ncomb = comb(n+degree-1,degree, exact=True)
    # Preallocating and start from all ones => x1*x1*x1
    monomials = np.ones((ncomb,degree), dtype=int)

    for i in range(1,ncomb):
        monomials[i] = monomials[i-1].copy()
        j = degree-1  # Index indicating which factor to change
        while monomials[i,j] == n:
            # Try to increase the last factor, but if this is not possible,
            # look the previous one that can be increased
            j -= 1
        # Increase factor j wrt previous monomial, e.g. x1*x1*x1 -> x1*x1*x2
        monomials[i,j] += 1
        # Monomial after x1*x1*xmax is x1*x2*x2, and not x1*x2*xmax
        monomials[i,j+1:degree] = monomials[i,j]

    # Exponents representation, e.g. [2, 1] represents x1^2*x2 = x1*x1*x2
    combinations = np.zeros((ncomb,n), dtype=int)
    for i in range(ncomb):  # # Loop over all terms
        for j in range(n):  # # Loop over all inputs
            # Count the number of appearances of input j in monomial i. +1 for
            # zero index
            combinations[i,j] = np.sum(monomials[i] == j+1)

    return combinations

def select_active(structure,n,m,q,nx):
    """Select active elements in E or F matrix.

    Select the active elements (i.e. those on which optimization will be done)
    in the E or F matrix. In particular, the linear indices (see also sub2ind
    and ind2sub) of the active elements in the transpose of the E or F matrix
    are calculated.

    Parameters:
    -----------
    structure: str
        string indicating which elements in the E or F matrix are active.
        'diagonal': active elements in row j of the E matrix are those
                    corresponding to pure nonlinear terms in state j (only for
                    state equation)
        'inputsonly' : only terms in inputs
        'statesonly' : only terms in states
        'nocrossprod' : no cross-terms
        'affine' : only terms that are linear in one state
        'affinefull' : only terms that are linear in one state or constant in
                       the states
        'full' : all terms
        'empty' : no terms
        'nolastinput' : no terms in last input
        row_E : only row given by row_E in E matrix is active (only for state
                equation)
    n : int
        number of states
    m : int
        number of inputs
    q : int
        number of rows in corresponding E/F matrix
           q = n if E matrix is considered,
           q = p if F matrix is considered
    nx : int | list
        degrees of nonlinearity in E/F matrix

    Returns
    -------
    active: linear indices of the active elements in the transpose of the E or
            F matrix

    Examples
    --------
    >>> n = 2  # Number of states
    >>> m = 1  # Number of inputs
    >>> p = 1  # Number of outputs
    >>> nx = 2 # Degree(s) of nonlinearity
    Powers of all possible terms in n+m inputs of degree(s) nx
    >>> terms = combinations(n+m,nx)
    array([[2, 0, 0],
           [1, 1, 0],
           [1, 0, 1],
           [0, 2, 0],
           [0, 1, 1],
           [0, 0, 2]])

    There are six quadratic terms in the two states x1 and x2, and the input u,
    namely x1^2, x1*x2, x1*u, x2^2, x2*u, and u^2. The matrix E is a 2 x 6
    matrix that contains the polynomial coefficients in each of these 6 terms
    for both state updates. The active elements will be calculated as linear
    indices in the transpose of E, hence E can be represented as
    E = [e1 e2 e3 e4  e5  e6;
         e7 e8 e9 e10 e11 e12]
    The matrix F is a 1 x 6 matrix that contains the polynomial coefficients in
    each of the 6 terms for the output equation. The matrix F can be
    represented as F = [f1 f2 f3 f4 f5 f6]

    **Diagonal structure**
    >>> activeE = select_active('diagonal',n,m,n,nx)
    array([0, 9])
    Only e1 and e10 are active. This corresponds to a term x₁² in the first
    state equation and a term x₂² in the second state equation.

    **Inputs only structure**
    >>> activeE = select_active('inputsonly',n,m,n,nx)
    array([ 5, 11])
    Only e6 and e12 are active. This corresponds to a term u² in both state
    equations. In all other terms, at least one of the states (possibly raised
    to a certain power) is a factor.

    >>> activeF = select_active('inputsonly',n,m,p,nx)
    array([5])
    Only f6 is active. This corresponds to a term u² in the output equation.

    **States only structure**
    >>> activeE = select_active('statesonly',n,m,n,nx)
    array([0, 1, 3, 6, 7, 9])

    Only e1, e2, e4, e7, e8, and e10 are active. This corresponds to terms x₁²,
    x₁*x₂, and x₂² in both state equations. In all other terms, the input
    (possibly raised to a certain power) is a factor.

    **No cross products structure**
    >>> activeE = select_active('nocrossprod',n,m,n,nx)
    array([ 0,  3,  5,  6,  9, 11])
    Only e1, e4, e6, e7, e10, and e12 are active. This corresponds to terms
    x₁², x₂², and u² in both state equations. All other terms are crossterms
    where more than one variable is present as a factor.

    **State affine structure**
    >>> activeE = select_active('affine',n,m,n,nx)
    array([ 2,  4,  8, 10])
    Only e3, e5, e9, and e11 are active. This corresponds to terms x₁*u and
    x₂*u in both state equations, since in these terms only one state appears,
    and it appears linearly.

    **Full state affine structure**
    >>> activeE = select_active('affinefull',n,m,n,nx)
    array([ 2,  4,  5,  8, 10, 11])
    Only e3, e5, e6, e9, e11, and e12 are active. This corresponds to terms
    x₁*u, x₂*u and u² in both state equations, since in these terms at most one
    state appears, and if it appears, it appears linearly.

    **Full structure**
    >>> activeE = select_active('full',n,m,n,nx)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
    All elements in the E matrix are active.

    **Empty structure**
    >>> activeE = select_active('empty',n,m,n,nx)
    array([], dtype=int64)
    None of the elements in the E matrix are active.

    ** One row in E matrix structure**
    row_E = 1  # Select which row in E is active
    >>> activeE = select_active(row_E,n,m,n,nx)
    array([ 6,  7,  8,  9, 10, 11])
    Only the elements in the second row of E are active

    **No terms in last input structure**
    This is useful in a PNLSS model when considering the initial state as a
    parameter. The state at time one can be estimated by adding an extra input
    u_art(t) that is equal to one at time zero and zero elsewhere. Like this,
    an extended PNLSS model is estimated, where the last column in its B matrix
    corresponds to the state at time one in the original PNLSS model. To ensure
    that the optimization is only carried out on the parameters of the original
    PNLSS model, only the corresponding coefficients in the E/F matrix should
    be selected as active.

    Powers of all possible terms with one extra input
    >>> terms_extended = combinations(n+m+1,nx)
    array([[2, 0, 0, 0],
           [1, 1, 0, 0],
           [1, 0, 1, 0],
           [1, 0, 0, 1],
           [0, 2, 0, 0],
           [0, 1, 1, 0],
           [0, 1, 0, 1],
           [0, 0, 2, 0],
           [0, 0, 1, 1],
           [0, 0, 0, 2]])
    The nonlinear terms in the extra input should not be considered for
    optimization.
    >>> activeE_extended = select_active('nolastinput',n,m+1,n,nx)
    array([ 0,  1,  2,  4,  5,  7, 10, 11, 12, 14, 15, 17])

    Only the terms where the last input is raised to a power zero are active.
    This corresponds to the elif structure == where all terms in the original
    PNLSS model are active.

    The example below illustrates how to combine a certain structure in the
    original model (e.g. 'nocrossprod') with the estimation of the initial
    state.

    >>> activeE_ext = select_active('nolastinput',n,m+1,n,nx)
    >>> activeE_ext = activeE_ext[select_active('nocrossprod',n,m,n,nx)]
    array([ 0,  4,  7, 10, 14, 17])

    This corresponds to the terms x₁², x₂², and u₁² in both rows of the
    E_extended matrix, and thus to all terms in the original model, except for
    the cross terms.

    Note that an alternative approach is to include the initial state
    in the parameter vector (TODO see also fLMnlssWeighted_x0u0).

    """
    # All possible nonlinear terms of degrees nx in n+m inputs
    combis = combinations(n+m,nx)
    n_nl = combis.shape[0]  # Number of terms

    stype = {'diagonal', 'inputsonly', 'statesonly', 'nocrossprod', 'affine',
             'affinefull', 'full', 'empty', 'nolastinput'}
    if structure == 'diagonal':
        # Diagonal structure requires as many rows in E (or F) matrix as the
        # number of states
        if n != q:
            raise ValueError('Diagonal structure can only be used in state'
                             ' equation, not in output equation')
        # Find terms that consist of one state, say x_j, raised to a nonzero
        # power
        active = np.where((np.sum(combis[:,:n] != 0,1) == 1) &
                          (np.sum(combis[:,n:] != 0,1) == 0))[0]
        # Select these terms only for row j in the E matrix
        for i, item in enumerate(active):
            # Which state variable is raised to a nonzero power
            tmp = np.where(combis[item] != 0)[0]
            # Linear index of active term in transpose of E
            active[i] += (tmp)*n_nl
    elif structure == 'inputsonly':
        # Find terms where all states are raised to a zero power
        active = np.where(np.sum(combis[:,:n] != 0,1) == 0)[0]
    elif structure == 'statesonly':
        # Find terms where all inputs are raised to a zero power
        active = np.where(np.sum(combis[:,n:] != 0,1) == 0)[0]
    elif structure == 'nocrossprod':
        # Find terms where only one variable (state or input) is raised to a
        # nonzero power
        active = np.where(np.sum(combis != 0,1) == 1)[0]
    elif structure == 'affine':
        # Find terms where only one state is raised to power one, and all
        # others to power zero. There are no conditions on the powers in the
        # input variables
        active = np.where((np.sum(combis[:,:n] != 0,1) == 1) &
                          (np.sum(combis[:,:n], 1) == 1))[0]

    elif structure == 'affinefull':
        # Find terms where at most one state is raised to power one, and
        # all others to power zero. There are no conditions on the powers
        # in the input variables
        active = np.where((np.sum(combis[:,:n] != 0,1) <= 1) &
                          (np.sum(combis[:,:n], 1) <= 1))[0]
    elif structure == 'full':
        # Select all terms in E/F matrix
        active = np.arange(q*n_nl)
    elif structure == 'empty':
        # Select no terms. We need to specify the array as int
        active = np.array([], dtype=int)
    elif structure == 'nolastinput':
        if m > 0:
            # Find terms where last input is raised to power zero
            active = np.where(combis[:,-1] == 0)[0]
        else:
            raise ValueError(f"There is no input for {structure}")
    else:
        # Check if one row in E is selected. Remember we use 0-based rows
        if (isinstance(structure, (int, np.integer)) and
            structure in np.arange(n)):
            row_E = int(structure)
            active = row_E*n_nl + np.arange(n_nl)
        else:
            raise ValueError(f"Wrong structure {structure}. Should be: {stype}"
                             f" or int specifying a row within 0-{n-1}")

    if structure in \
       ('inputsonly','statesonly','nocrossprod','affine','affinefull',
        'nolastinput'):
        # Select terms for all rows in E/F matrix
        active = (np.tile(active[:,None], q) +
                  np.tile(np.linspace(0,(q-1)*n_nl,q, dtype=int),
                          (len(active),1))).ravel()

    # Sort the active elements
    return np.sort(active)

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
                         **system.xpowers, axis=1)
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]) +
                        np.dot(system.E, zeta_t))
        # Output equation y(t) = C*x(t) + D*u(t) + F*eta(x(t),u(t))
        eta_t = np.prod(np.outer(repmat_y, np.hstack((xout[i], u_dt[i])))
                        **system.ypowers, axis=1)
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]) +
                      np.dot(system.F, eta_t))

    # Last point
    eta_t = np.prod(np.outer(repmat_y, np.hstack((xout[-1], u_dt[-1])))
                    **system.ypowers, axis=1)
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

    out = np.zeros((p,N,nactive))
    for k, activ in enumerate(active):
        # Which column in A, B, or E matrix
        j = np.mod(activ, npar)
        # Which row in A, B, or E matrix
        i = (activ-j)//npar
        # partial derivative of x(0) wrt. A(i,j), B(i,j), or E(i,j)
        Jprev = np.zeros(n)
        for t in range(1,N):
            # Calculate state update alternative state-space model at time t
            # Terms in alternative states at time t-1
            J = A_Edwdx[:,:,t-1] @ Jprev
            # Term in alternative input at time t-1
            J[i] += samples[t-1,j]
            # Calculate output alternative state-space model at time t
            out[:,t,k] = C_Fdwdx[:,:,t] @ J
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
        FdwyIdx = np.zeros(shape=(*A.shape,n_trans))
    else:
        A_EdwxIdx = multEdwdx(contrib,system.xd_powers,np.squeeze(system.xd_coeff),
                          E,n)
    A_EdwxIdx += A[...,None]
    zeta = nl_terms(contrib, system.xpowers).T  # (NT,n_nx)

    # F∂ₓη  (p,n,NT)
    if F.size == 0:
        FdwyIdx = np.zeros(shape=(*C.shape,n_trans))
    else:
        FdwyIdx = multEdwdx(contrib,system.yd_powers,np.squeeze(system.yd_coeff),
                  F,n)
    # Add C to F∂ₓη for all samples at once
    FdwyIdx += C[...,None]
    eta = nl_terms(contrib, system.ypowers).T  # (NT,n_ny)

    # calculate jacobians wrt state space matrices
    JC = np.kron(np.eye(p), system.x_mod)  # (p*N,p*n)
    JD = np.kron(np.eye(p), system.signal.um)  # (p*N, p*m)
    if system.yactive.size:
        JF = np.kron(np.eye(p), eta)  # Jacobian wrt all elements in F
        JF = JF[:,system.yactive]  # all active elements in F. (p*NT,nactiveF)
        JF = JF[system.idx_remtrans]  # (p*N,nactiveF)
    else:
        JF = np.array([]).reshape(p*N,0)

    # calculate Jacobian by filtering an alternative state-space model
    JA = element_jacobian(x_trans, A_EdwxIdx, FdwyIdx, np.arange(n**2))
    JA = JA.transpose((1,0,2)).reshape((p*n_trans, n**2))
    JA = JA[system.idx_remtrans]  # (p*N,n**2)

    JB = element_jacobian(u_trans, A_EdwxIdx, FdwyIdx, np.arange(n*m))
    JB = JB.transpose((1,0,2)).reshape((p*n_trans, n*m))
    JB = JB[system.idx_remtrans]  # (p*N,n*m)

    if system.xactive.size:
        JE = element_jacobian(zeta, A_EdwxIdx, FdwyIdx, system.xactive)
        JE = JE.transpose((1,0,2)).reshape((p*n_trans, len(system.xactive)))
        JE = JE[system.idx_remtrans]  # (p*N,nactiveE)
    else:
        JE = np.array([]).reshape(p*N,0)

    jac = np.hstack((JA, JB, JC, JD, JE, JF))[without_T2]
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
