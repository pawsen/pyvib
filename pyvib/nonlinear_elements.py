#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from pyvib.polynomial import combinations, nl_terms, poly_deriv, select_active


"""Nonlinear functions
"""


class Nonlinear_Element:
    """Bare class
    """
    pass


class Pnlss(Nonlinear_Element):
    """Combinations of monomials in x and u"""

    def __init__(self, eq, degree, structure):
        """Initialize active nonlinear terms/monomials to be optimized"""
        self.eq = eq
        self.degree = np.asarray(degree)
        self.structure = structure
        self.n_ny = 0
            
    @property
    def active(self):
        return self._active
    
    def set_active(self,n,m,p,q):
        # all possible terms
        self.powers = combinations(n+m, self.degree)
        self.n_nx = self.powers.shape[0]
        self.n_nl = self.n_nx
        
        # This is not needed. We can use q. q = n for 'x' and q = p for 'y'
        if self.eq in ('state', 'x'):
            active = select_active(self.structure,n,m,n,self.degree)
        if self.eq in ('output', 'y'):
            active = select_active(self.structure,n,m,p,self.degree)
        
        self._active = active
        # Compute the derivatives of the polynomials
        self.d_powers, self.d_coeff = poly_deriv(self.powers)
        
    def fnl(self,x,y,u):
        # TODO: Ohh, code SUCKS!!!
        repmat = np.ones(self.n_nx)  # maybe init in set_active? save time?
        ndim = x.ndim
        ns = 1
        if ndim > 1:
            ns = x.shape[0]
        if ns == 1:
            zeta = np.prod(np.outer(repmat, np.hstack((x, u)))**self.powers,axis=1)
        else:
            # same as nl_terms(signal,power).
            zeta = np.prod(((np.kron(repmat, np.hstack((x, u)).T[None].T)).T)**self.powers[:,:,None],axis=1)
         
        return zeta
    
    def dfdy(self,x,y,u):
        """Derivative wrt output, thus zero"""
        return np.array([])
    
    def dfdx(self,x,y,u):
        n = len(x)  # len returns 1dim(x.shape[0]) and allows x to be list.
        signal = np.atleast_2d(np.hstack((x, u))).T
        # n_all = number of signals x and u; ns = number of samples
        n_all, ns = signal.shape
        # n_nx = number of monomials in w
        n_nx = self.n_nx
        dfdx = np.zeros((n_nx,n,ns))
        # Loop over all signals x w.r.t. which derivatives are taken
        for k in range(n):
            # Repeat coefficients of derivative of w w.r.t. x_k
            A = np.outer(self.d_coeff[:,k], np.ones(ns))
            for j in range(n_all):     # Loop over all signals x and u
                for i in range(n_nx):  # Loop over all monomials
                    # Derivative of monomial i wrt x_k
                    A[i,:] *= signal[j,:]**self.d_powers[i,j,k]
            dfdx[:,k,:] = A
    
        return dfdx
    
    def df(x,u):
        """Example of generic derivative. Could be like this:
        
        def dfdy(...):
            if self.eq in (...):
                return df(some args)
            else:
                return df(oher args)
        """
        pass

#from pyvib.polynomial import multEdwdx

#pnlss = Pnlss('x', 2, 'full')    
#pnlss.set_active(2,1,1,2)
#
#x = np.array([2,1])
#u = np.array([1])
#
#dhdy = pnlss.dfdx(x,1,u)
#signal = np.atleast_2d(np.hstack((x, u))).T
#n = 2
#E = np.arange(n*pnlss.n_nx).reshape(n,pnlss.n_nx)
#
#Edx = multEdwdx(signal, pnlss.d_powers, pnlss.d_coeff, E, n)
#
#Gdidy = np.einsum('ij,jkl->ikl',E,dhdy)


class Polynomial_x(Nonlinear_Element):
    def __init__(self, exponent, w):
        self.w = np.atleast_2d(w)
        self.exponent = np.atleast_1d(exponent)
        # number of nonlinear elements
        self.n_nx = 1
        self.n_ny = 0
        self.n_nl = self.n_nx + self.n_ny
        self._active = np.array([],dtype=np.intp)
        
    @property
    def active(self):
        return self._active
        
    def set_active(self,n,m,p,q):
        # all are active
        self._active = np.s_[0:q*self.n_nl]

    def fnl(self, x,y,u):
        w = self.w
        x = np.atleast_1d(x)
        # displacement of dofs attached to nl
        xnl = np.atleast_2d(np.inner(w, x))  # (n_nx, ns)
        f = np.prod(xnl.T**self.exponent, axis=1)
        return f

    def dfdy(self,x,y,u):
        """Output nl, thus zero"""
        return np.array([])

    def dfdx(self,x,y,u):
        w = self.w
        x = np.atleast_1d(x)
        xnl = np.inner(w, x)
        # same as np.outer when we do w.T
        dfdx = self.exponent[:,None] * xnl**(self.exponent-1) * w.T # (n, ns)
        return dfdx


class Polynomial(Nonlinear_Element):
    """Polynomial output nonlinearity
    """

    def __init__(self, exponent, w, structure='Full'):
        """
        exponents: ndarray (n_ny)
        w: ndarray (n_ny, p)

        Example
        -------
        fnl = y₁ẏ₁, y = [y₁, y₂, ẏ₁, ẏ₂]
        exponents = [1,1]
        w = np.array([[1,0,0,0], [0,0,1,0]])
        """
        self.w = np.atleast_2d(w)
        self.exponent = np.atleast_1d(exponent)
        self.structure = structure
        # number of nonlinear elements
        self.n_nx = 0
        self.n_ny = 1
        self.n_nl = self.n_nx + self.n_ny
        self._active = np.array([],dtype=np.intp)
        
    @property
    def active(self):
        """Select active part of E"""
        return self._active
        
    def set_active(self,n,m,p,q):
        # all are active
        #pass
        self._active = np.r_[0:q*self.n_nl]

    def fnl(self, x,y,u):
        """
        x: ndarray (n, ns)
        y: ndarray (p, ns)

        Return
        ------
        f: ndarray (ns,)
        """
        w = self.w
        y = np.atleast_1d(y)
        # displacement of dofs attached to nl
        ynl = np.atleast_2d(np.inner(w, y)) # maybe y.T  # (n_ny, ns)
        f = np.prod(ynl.T**self.exponent, axis=1)

        return f

    def dfdx(self,x,y,u):
        """Output nl, thus zero"""
        return np.array([])

    def dfdy(self,x,y,u):
        """
        TODO: only works for single output, ex. y₁³
        dfdy (p,ns)  # should be (p, n_nx, ns)
        """
        w = self.w
        y = np.atleast_1d(y)
        ynl = np.inner(w, y)
        # same as np.outer when we do w.T
        dfdy = self.exponent[:,None] * ynl**(self.exponent-1) * w.T # (p, ns)
        return dfdy


class NLS(object):
    """ Assemble nonlinear attachments
    
    Note that each attachment is copied to ensure it truly belong to the NLS 
    it was initiated with
    
    """

    def __init__(self, nls=None):
        self.nls = []
        self.n_nx = 0
        self.n_ny = 0
        self.n_nl = 0
        # different indexes needed to keep track of which par is x- and
        # y-dependent. ugly.
        self.idx = np.array([],dtype=np.intp)
        self.idy = np.array([],dtype=np.intp)
        self.active = np.array([],dtype=np.intp)
        self.xactive = np.array([],dtype=np.intp)
        self.yactive = np.array([],dtype=np.intp)
        self.jac_x = np.array([],dtype=np.intp)
        self.jac_y = np.array([],dtype=np.intp)
        if nls is not None:
            self.add(nls)

    def add(self, nls):
        if not isinstance(nls, list):
            nls = [nls]
        for nl in nls:
            self.nls.append(deepcopy(nl))  # copy!

    def set_active(self,m,n,p,q):
        """Select active part of E
        n,m,p : int
            number of states, inputs, outputs
        q : int
            number of rows in corresponding E/F matrix
                q = n if E matrix is considered,
                q = p if F matrix is considered
        """
        # This is a very ugly piece of code. All this is needed for splitting 
        # E in E/G, where E are coefficients for x-dependen nl's and G for y.
        # And recombining the jacobians JE/JG

        # first we need to determine the total number of NLs. This can only be
        # done when the system size (n,m,p) is known
        for nl in self.nls:
            # Find columns in E(n,n_nl) matrix which correspond to x-dependent
            # nl (idx) and y-dependent (idy). Note it does not matter if we
            # have '+nl.n_nx' in self.idy or '+nl.n_ny' in self.idx
            self.idx = np.r_[self.idx, self.n_nl         + np.r_[0:nl.n_nx]]
            self.idy = np.r_[self.idy, self.n_nl+nl.n_nx + np.r_[0:nl.n_ny]]
            self.n_nx += int(nl.n_nx)
            self.n_ny += int(nl.n_ny)
            self.n_nl += int(nl.n_nl)

        # If the same NLS object is used multiple times, we need to
        # reset the active count, as done here.
        self.active = np.array([],dtype=np.intp)
        self.jac_active = np.array([],dtype=np.intp)
        self.xactive = np.array([],dtype=np.intp)
        self.yactive = np.array([],dtype=np.intp)
        n_nl = 0
        for nl in self.nls:
            nl.set_active(m,n,p,q)
            # convert local index to global index. Active elements in global E
            npar = nl.n_nl
            active = np.r_[nl.active]  # active might be a slice -> convert
            col = np.mod(active, npar)
            row = (active-col)// npar
            idx = row*self.n_nl + col
            self.active = np.r_[self.active, n_nl + idx]
            n_nl += npar

        # get permution index for combining JE and JG. We need this so
        # θ(flattened parameters) correspond to the right place in the jacobian
        nlj = 0
        self.jac_x = np.array([],dtype=np.intp)
        self.jac_y = np.array([],dtype=np.intp)
        for nl in self.nls:
            active = np.r_[nl.active]
            self.jac_x = np.r_[self.jac_x, nlj                       +\
                          np.r_[:nl.n_nx*len(active)]]
            self.jac_y = np.r_[self.jac_y, nlj + nl.n_nx*len(active) +\
                          np.r_[:nl.n_ny*len(active)]]
            nlj += nl.n_nl*len(active)

        # get (local) index in splitted E/G matrix
        col = np.mod(self.active,self.n_nl)
        row = (self.active-col)//self.n_nl
        for i, lcol in enumerate(self.idx):
            idx = col == lcol
            tmp = row[idx]*len(self.idx) + i
            self.xactive = np.r_[self.xactive, tmp]
            
        for i, lcol in enumerate(self.idy):
            idx = col == lcol
            tmp = row[idx]*len(self.idy) + i
            self.yactive = np.r_[self.yactive, tmp]


    def fnl(self,x,y,u):
        """
        Returns
        -------
        fnl: ndarray(n_nl, ns)
            If ns = 1, returns 1d array
        """

        # return empty array in case of no nonlinearities
        if len(self.nls) == 0:
            return np.array([])

        y = np.atleast_2d(y)
        ns = y.shape[0]
        if ns == 1:  # TODO another hack!!!
            fnl = np.empty((self.n_nl))
        else:
            fnl = np.empty((self.n_nl,ns))
        
        nls = 0
        for i, nl in enumerate(self.nls):
            n_nl = nl.n_nl
            fnl[nls:nls+n_nl] = nl.fnl(x,y,u)
            nls += n_nl

        # remove last dim if ns = 1
        return fnl #.squeeze(-1)
    
    def dfdy(self,x,y,u):
        """
        Returns
        -------
        dfdx: ndarray (n_ny,p,ns)
           If ns=1, returns 2d array
        """
        # return empty array in case of no nonlinearities
        if self.n_ny == 0:
            return np.array([])

        y = np.atleast_2d(y)
        ns,p = y.shape
        dfdy = np.empty((self.n_ny,p,ns))

        nls = 0
        for nl in self.nls:
            tmp = nl.dfdy(x,y,u)
            if tmp.size:
                n_ny = nl.n_ny
                dfdy[nls:nls+n_ny] = tmp
                nls += n_ny
        
        return dfdy

    def dfdx(self,x,y,u):
        """
        Returns
        -------
        dfdx: ndarray (n_nx,n,ns)
           If ns=1, returns 2d array
        """
        if self.n_nx == 0:
            return np.array([])

        x = np.atleast_2d(x)
        ns,n = x.shape
        dfdx = np.empty((self.n_nx,n,ns))

        nls = 0
        for nl in self.nls:
            tmp = nl.dfdx(x,y,u)
            if tmp.size:
                n_nx = nl.n_nx
                dfdx[nls:nls+n_nx] = tmp
                nls += n_nx
        
        return dfdx



#exponent = np.array([3])
#w = [1,0]
#poly1 = Polynomial(exponent,w)
#poly2 = Polynomial(exponent*2,w)
#nls = NLS([poly1, poly2])
#
#y1 = 3
#y2 = np.array([[3],[3]])
#y = np.arange(2*10).reshape(2,10)
#u = 0
#x = np.arange(3*10).reshape(3,10)
##print(nls.fnl(2,y1,1))
#print(nls.fnl(2,y2,1))
#print(nls.fnl(x,y,u))
#print('dfdy')
#print(nls.dfdy(2,y,1))
#direction =

#exponents = [2]
#exponents = np.array(exponents)
#w = [1,0,0]
#w = np.atleast_2d(w)
#y = np.arange(3*10).reshape((3,10))
#ynl = np.inner(w, y.T)
#f = np.prod(ynl.T**exponents, axis=1)
#
#print(dfdy)
#
#exponents = [2,2]
#exponents = np.array(exponents)
#w = np.array([[1,0,0],[0,0,1]])
#w = np.atleast_2d(w)
#y = np.arange(3*10).reshape((3,10))
#ynl = np.inner(w, y.T)
#f = np.prod(ynl.T**exponents, axis=1)
#dfdy = exponents[:,None] * ynl**(exponents-1)
#
#print(dfdy*w.T)


#x = np.arange(3*100).reshape((3,100))/0.01
