#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import comb

"""Routines for creating polynomial representation

"""

def nl_terms(signal,power):
    # NB THIS ONE IS NOT NEEDED! DELETE SOON!
    """Construct polynomial terms.

    Computes polynomial terms, where contrib contains the input signals to the
    polynomial and pow contains the exponents of each term in each of the
    inputs.

    Parameters
    ----------
    signal : ndarray(n+m,nt)
        array with nt samples of the input signals to the polynomial.
        Typically, these are the n states and the m inputs of the nonlinear
        state-space model.
    power : ndarray(nterms,n+m)
        matrix with the exponents of each term in each of the inputs to the
        polynomial

    Returns
    -------
    out : ndarray(nterms,nt)
        matrix with nt samples of each term

    Examples
    --------
    >>> n = 2  # Number of states
    >>> m = 1  # Number of inputs
    >>> nt = 1000  # Number of samples
    >>> x = np.random.randn(nt,n)  # States
    >>> u = np.random.randn(nt,m)  # Input
    >>> contrib = np.hstack((x, u)).T  # States and input combined
    All possible quadratic terms in states and input: x1^2, x1*x2, x1*u, x2^2,
    x2*u, u^2
    >>> pow = np.array([[2,0,0],
                        [1,1,0],
                        [1,0,1],
                        [0,2,0],
                        [0,1,1],
                        [0,0,2]])
    >>> nl_terms(contrib,pow)
    array([x[:,0]**2,
           x[:,0]*x[:,1],
           x[:,0]*u.squeeze(),
           x[:,1]**2,
           x[:,1]*u.squeeze(),
           u.squeeze()**2])
    """

    # Number of samples
    N = signal.shape[1]
    # Number of terms
    nterms = power.shape[0]
    out = np.empty((nterms,N))
    for i in range(nterms):
        # All samples of term i
        out[i] = np.prod(signal**power.T[:,None,i], axis=0)

    return out




def poly_deriv(powers):
    """Calculate derivative of a multivariate polynomial
    """
    # Polynomial coefficients of the derivative
    d_coeff = powers
    n = powers.shape[1]
    #  Terms of the derivative
    d_powers = np.repeat(powers[...,None],n, axis=2)
    for i in range(n):
        # Derivative w.r.t. variable i has one degree less in variable i than
        # original polynomial If original polynomial is constant w.r.t.
        # variable i, then the derivative is zero, but take abs to avoid a
        # power -1 (zero coefficient anyway)
        d_powers[:,i,i] = np.abs(powers[:,i]-1)

        # TODO
        # This would be more correct, but is slower
        # d_powers(:,i,i) = powers(:,i) - 1;
        # d_powers(powers(:,i) == 0,:,i) = 0;

    return d_powers, d_coeff


def multEdwdx(signal, power, coeff, E, n):
    """Multiply a matrix E with the derivative of a polynomial w(x,u) wrt. x

    Multiplies a matrix E with the derivative of a polynomial w(x,u) wrt the n
    elements in x. The samples of x and u are given by `contrib`. The
    derivative of w(x,u) wrt. x, is given by the exponents in x and u (given in
    power) and the corresponding coefficients (given in coeff).

    Parameters
    ----------
    signal : ndarray(n+m,N)
        N samples of the signals x and u
    power : ndarray(n_nx,n+m,n+m)
        The exponents of the derivatives of w(x,u) w.r.t. x and u, i.e.
        power(i,j,k) contains the exponent of contrib j in the derivative of
        the ith monomial w.r.t. contrib k.
    coeff : ndarray(n_nx,n+m)
        The corresponding coefficients, i.e. coeff(i,k) contains the
        coefficient of the derivative of the ith monomial in w(x,u) w.r.t.
        contrib k.
    E : ndarray(n_out,n_nx)
    n : int
        number of x signals w.r.t. which derivatives are taken

    Returns
    -------
    out : ndarray(n_out,n,N)
        Product of E and the derivative of the polynomial w(x,u) w.r.t. the
        elements in x at all samples.

    Examples
    --------
    Consider w(x1,x2,u) = [x1^2    and E = [1 3 5
                           x1*x2            2 4 6]
                           x2*u^2]
    then the derivatives of E*w wrt. x1 and x2 are given by
    E*[2*x1 0
       1*x2 1*x1
       0    1*u^2]
    and the derivative of w wrt. u is given by [0,0,2*x2*u]^T

    >>> E = np.array([[1,3,5],[2,4,6]])
    >>> pow = np.zeros((3,3,3))
    Derivative wrt. x1 has terms 2*x1, 1*x2 and 0
    >>> pow[:,:,0] = np.array([[1,0,0],
                               [0,1,0],
                               [0,0,0]])
    Derivative wrt. x2 has terms 0, 1*x1 and 1*u^2
    >>> pow[:,:,1] = np.array([[0,0,0],
                               [1,0,0],
                               [0,0,2]])
    Derivative wrt. u has terms 0, 0 and 2*x2*u
    >>> pow[:,:,2] = np.array([[0,0,0],
                               [0,0,0],
                               [0,1,1]])
    >>> coeff = np.array([[2,0,0],
                          [1,1,0],
                          [0,1,2]])
    >>> n = 2  # Two signals x
    Ten random samples of signals x1, x2, and u
    >>> contrib = np.random.randn(3,10)
    >>> out = multEdwdx(contrib,pow,coeff,E,n)
    >>> t = 0
    out[:,:,t] = E @ np.array([[2*contrib[0,t],0],
                              [1*contrib[1,t],1*contrib[0,t]],
                              [0             ,1*contrib[2,t]**2]])

    """
    # n_all = number of signals x and u; nt = number of samples
    n_all, nt = signal.shape
    # n_out = number of rows in E; n_nx = number of monomials in w
    n_out, n_nx = E.shape
    out = np.zeros((n_out,n,nt))
    # Loop over all signals x w.r.t. which derivatives are taken
    for k in range(n):
        # Repeat coefficients of derivative of w w.r.t. x_k
        A = np.outer(coeff[:,k], np.ones(nt))
        for j in range(n_all):     # Loop over all signals x and u
            for i in range(n_nx):  # Loop over all monomials
                # Derivative of monomial i wrt x_k
                A[i,:] *= signal[j,:]**power[i,j,k]
        # E times derivative of w wrt x_k
        out[:,k,:] = np.matmul(E,A)

    return out


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
    
    Parameters
    ----------
    n: int
        Number of variables, eg x1, x2, ...
    degree: int
    
    Return
    ------
    combinations: ndarray(ncomb,n)

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
