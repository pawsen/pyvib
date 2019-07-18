#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Nonlinear_Element:
    pass

class pol:

    def __init__(self, exponents, w):
        """
        exponents: ndarray (n_ny)
        w: ndarray (n_ny, p)

        Example
        -------
        fnl = y₁ẏ₁, y = [y₁, y₂, ẏ₁, ẏ₂]
        exponents = [1,1]
        w = np.array([[1,0,0,0], [0,0,1,0]])
        """
        self.w = w
        self.exponents = np.array(exponents)

    def fnl(self, x,y,u):
        """
        x: ndarray (n, ns)
        y: ndarray (p, ns)

        Return
        ------
        f: ndarray (ns,)
        """

        w = self.w
        # displacement of dofs attached to nl
        ynl = np.inner(w, y.T)  # (n_ny, ns)
        f = np.prod(ynl.T**self.exponents, axis=1)

        return f

    def dfdx(self,x,y,u):
        """Output nl, thus zero"""
        return 0

    def dfdy(self,x,u,y):
        """
        TODO: only works for single output, ex. y₁³
        dfdy (p,ns)
        """
        w = self.w

        ynl = np.inner(w, y.T)

        exponents[:,None] * ynl
        return

    # def _poly_deriv(self):
    #     """Find the derivatives of the given exponents"""


#direction =

exponents = [2]
exponents = np.array(exponents)
w = [1,0,0]
w = np.atleast_2d(w)
y = np.arange(3*10).reshape((3,10))
ynl = np.inner(w, y.T)
f = np.prod(ynl.T**exponents, axis=1)
dfdy = exponents[:,None] * ynl**(exponents-1) * w.T  # (p, ns)
print(dfdy)

exponents = [2,2]
exponents = np.array(exponents)
w = np.array([[1,0,0],[0,0,1]])
w = np.atleast_2d(w)
y = np.arange(3*10).reshape((3,10))
ynl = np.inner(w, y.T)
f = np.prod(ynl.T**exponents, axis=1)
dfdy = exponents[:,None] * ynl**(exponents-1)
print(dfdy*w.T)


#x = np.arange(3*100).reshape((3,100))/0.01
