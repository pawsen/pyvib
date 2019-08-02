#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np


"""Nonlinear functions to simulate: using :class:`.newmark.Newmark`.

New nonlinear elements should derive :class:`Nonlinear_Element` and implement
::
  def fnl(q,u)           # nonlinear force
  def dfdq(q,u)          # derivative wrt position
  def dfdu(q,u)          # derivative wrt velocity
"""


class Nonlinear_Element:
    """Bare class
    """

    def __init__(self, **kwargs):
        # only initialize things if they are not already set
        super().__init__(**kwargs)


class NLS(object):
    """ Assemble nonlinear attachments for usage with :class:`.newmark.Newmark`

    Note that each attachment is copied to ensure it truly belong to the NLS
    it was initiated with

    For identification, see :class:`.nlss.NLSS`
    """

    def __init__(self, nls=None):
        self.nls = []
        self.n_nl = 0
        if nls is not None:
            self.add(nls)

    def add(self, nls):
        if not isinstance(nls, list):
            nls = [nls]
        for nl in nls:
            self.nls.append(deepcopy(nl))

    def fnl(self, q, u):
        """Nonlinear force

        q: ndarray(ndof, ns), displacement
        u: ndarray(ndof, ns), velocity

        Returns
        -------
        fnl: ndarray(ndof, ns)
            If ns = 1, returns 1d array
        """
        # return zero array in case of no nonlinearities
        if len(self.nls) == 0:
            return np.zeros(len(q))
        ndof = len(q)
        fnl = np.zeros(ndof)
        for nl in self.nls:
            fnl += nl.fnl(q, u)
        return fnl

    def dfnl(self, q, u):
        """Derivative of nonlinear force wrt. `q` and `u`

        q: ndarray(ndof, ns), displacement
        u: ndarray(ndof, ns), velocity

        Returns
        -------
        dfnl_dq: ndarray (ndof,ndof,ns)
           If ns=1, returns 2d array
        dfnl_du: ndarray (ndof,ndof,ns)
           If ns=1, returns 2d array
        """
        # return zero array in case of no nonlinearities
        if len(self.nls) == 0:
            return np.zeros((len(q), len(q)))

#        q = np.atleast_2d(q)
#        ndof, ns = q.shape
        ndof = len(q)
        dfnl_dq = np.zeros((ndof, ndof))
        dfnl_du = np.zeros((ndof, ndof))
        for nl in self.nls:
            dfnl_dq += nl.dfnl_dq(q, u)
            dfnl_du += nl.dfnl_du(q, u)

        return dfnl_dq, dfnl_du


class Tanhdryfriction(Nonlinear_Element):
    """Regularized friction model.

    `f = kt*tanh(ẏ/eps)`

    sign(ẏ) approximated by tanh. eps control the slope. kt is the
    friction limit force.
    """

    def __init__(self, eps, w, kt=1, **kwargs):
        self.eps = eps
        self.w = np.atleast_1d(w)
        self.kt = kt
        super().__init__(**kwargs)

    def fnl(self, q, u):
        # displacement of dofs attached to nl
        unl = np.inner(self.w, u)
        f = np.outer(self.w, self.kt*np.tanh(unl / self.eps))
        return f.ravel()

    def dfnl_dq(self, q, u):
        return np.zeros((len(q), len(q)))

    def dfnl_du(self, q, u):
        unl = np.inner(self.w, u)
        dfnl_du = np.outer(self.w,
                           self.kt*(1 - np.tanh(unl / self.eps)**2) / self.eps
                           * self.w)
        return dfnl_du


class Polynomial(Nonlinear_Element):
    """Polynomial output nonlinearity

    Example
    -------
    fnl = (y₁-y₂)ẏ₁, where y = [y₁, y₂, ẏ₁, ẏ₂]
    exponents = [1,1]
    w = np.array([[1,-1,0,0], [0,1,0,0]]
    """

    def __init__(self, exp, w, k=1, **kwargs):
        """
        exponents: ndarray (n_ny)
        w: ndarray (n_ny, p)
        """
        self.w = np.atleast_1d(w)
        self.exp = np.atleast_1d(exp)
        self.k = k

    def fnl(self, q, u):
        qnl = np.inner(self.w, q)
        f = np.outer(self.w, self.k * qnl**self.exp)
        return f.ravel()

    def dfnl_du(self, q, u):
        """Displacement nl, thus zero"""
        return np.zeros((len(q), len(q)))

    def dfnl_dq(self, q, u):
        """
        dfdy (1, p, ns)
        """
        qnl = np.inner(self.w, q)
        dfnl_dq = np.outer(self.w, self.exp*qnl**(self.exp-1) * self.w)

        return dfnl_dq
