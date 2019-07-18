#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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

def multEdwdx(contrib, power, coeff, E, n):
    """Multiply a matrix E with the derivative of a polynomial w(x,u) wrt. x

    Multiplies a matrix E with the derivative of a polynomial w(x,u) wrt the n
    elements in x. The samples of x and u are given by `contrib`. The
    derivative of w(x,u) wrt. x, is given by the exponents in x and u (given in
    power) and the corresponding coefficients (given in coeff).

    Parameters
    ----------
    contrib : ndarray(n+m,N)
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
    # n_all = number of signals x and u; N = number of samples
    n_all, N = contrib.shape
    # n_out = number of rows in E; n_nx = number of monomials in w
    n_out, n_nx = E.shape
    out = np.zeros((n_out,n,N))
    # Loop over all signals x w.r.t. which derivatives are taken
    for k in range(n):
        # Repeat coefficients of derivative of w w.r.t. x_k
        A = np.outer(coeff[:,k], np.ones(N))
        for j in range(n_all):     # Loop over all signals x and u
            for i in range(n_nx):  # Loop over all monomials
                # Derivative of monomial i wrt x_k
                A[i,:] *= contrib[j,:]**power[i,j,k]
        # E times derivative of w wrt x_k
        out[:,k,:] = np.matmul(E,A)

    return out

def nl_terms(contrib,power):
    """Construct polynomial terms.

    Computes polynomial terms, where contrib contains the input signals to the
    polynomial and pow contains the exponents of each term in each of the
    inputs. The maximum degree of an individual input is given in max_degree.

    Parameters
    ----------
    contrib : ndarray(n+m,N)
        matrix with N samples of the input signals to the polynomial.
        Typically, these are the n states and the m inputs of the nonlinear
        state-space model.
    power : ndarray(nterms,n+m)
        matrix with the exponents of each term in each of the inputs to the
        polynomial

    Returns
    -------
    out : ndarray(nterms,N)
        matrix with N samples of each term

    Examples
    --------
    >>> n = 2  # Number of states
    >>> m = 1  # Number of inputs
    >>> N = 1000  # Number of samples
    >>> x = np.random.randn(N,n)  # States
    >>> u = np.random.randn(N,m)  # Input
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
    N = contrib.shape[1]
    # Number of terms
    nterms = power.shape[0]
    out = np.empty((nterms,N))
    for i in range(nterms):
        # All samples of term i
        out[i] = np.prod(contrib**power.T[:,None,i], axis=0)

    return out


class NL_force(object):

    def __init__(self, nls=None):
        self.nls = []
        if nls is not None:
            self.add(nls)

    def add(self, nls):
        if not isinstance(nls, list):
            nls = [nls]
            for nl in nls:
                self.nls.append(nl)

    def force(self, x, xd):

        # return empty array in case of no nonlinearities
        if len(self.nls) == 0:
            return np.array([])

        fnl = []
        for nl in self.nls:
            fnl_t = nl.compute(x, xd)
            fnl.extend(fnl_t)

        fnl = np.asarray(fnl)
        return fnl

class NL_polynomial():
    """Calculate force contribution for polynomial nonlinear stiffness or
    damping, see eq(2)

    Parameters
    ----------
    x : ndarray (ndof, ns)
        displacement or velocity.
    inl : ndarray (nbln, 2)
        Matrix with the locations of the nonlinearities,
        ex: inl = np.array([[7,0],[7,0]])
    enl : ndarray
        List of exponents of nonlinearity
    knl : ndarray (nbln)
        Array with nonlinear coefficients. ex. [1,1]
    idof : ndarray
        Array with node mapping for x.

    Returns
    -------
    f_nl : ndarray (nbln, ns)
        Nonlinear force
    """

    def __init__(self, inl, enl, knl, is_force=True):
        self.inl = inl
        self.enl = enl
        self.knl = knl
        self.is_force = is_force
        self.nnl = inl.shape[0]

    def compute(self, x, xd):
        inl = self.inl
        nbln = inl.shape[0]
        if self.is_force is False:
            # TODO: Overskriver dette x i original funktion? Ie pass by ref?
            x = xd

        ndof, nsper = x.shape
        idof = np.arange(ndof)
        fnl = np.zeros((nbln, nsper))

        for j in range(nbln):
            # connected from
            i1 = inl[j,0]
            # conencted to
            i2 = inl[j,1]

            # Convert to the right index
            idx1 = np.where(i1==idof)
            # if connected to ground
            if i2 == -1:
                x12 = x[idx1]
            else:
                idx2 = np.where(i2==idof)
                x12 = x[idx1] - x[idx2]
            # in case of even functional
            if (self.enl[j] % 2 == 0):
                x12 = np.abs(x12)

            fnl[j,:] = self.knl[j] * x12**self.enl[j]

        return fnl

class NL_spline():
    from .spline import spline

    def __init__(self, inl, nspl, is_force=True):
        self.nspline = nspl
        self.is_force = is_force
        self.inl = inl

        # number of nonlinearities * number of knots
        self.nnl = inl.shape[0]*(nspl+1)

    def compute(self, x, xd):
        inl = self.inl
        nbln = inl.shape[0]
        ndof, nsper = x.shape
        idof = np.arange(ndof)
        if self.is_force is False:
            x = xd

        fnl = []
        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1==idof)
            # if connected to ground
            if i2 == -1:
                x12 = x[idx1]
            else:
                idx2 = np.where(i2==idof)
                x12 = x[idx1] - x[idx2]
            fnl_t, kn, dx = spline(x12.squeeze(), self.nspline)

            fnl.extend(fnl_t)
        fnl = np.asarray(fnl)

        self.kn = kn
        self.fnl = fnl

        return fnl
