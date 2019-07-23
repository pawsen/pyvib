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
    def __init__(self, **kwargs):
        # only initialize things if they are not already set
        if not hasattr(self, 'n_nx'):
            self.n_nx = 0
        if not hasattr(self, 'n_ny'):
            self.n_ny = 0
        if not hasattr(self, '_active'):
            self._active = np.array([],dtype=np.intp)
        super().__init__(**kwargs)
        
    @property
    def n_nl(self):
        """Total number of nonlinear functions in class"""
        return self.n_nx + self.n_ny
    
    @property
    def active(self):
        """Returns active part of E"""
        return self._active
    
class Unilatteralspring(Nonlinear_Element):
    """Unilatteral spring. Estimate the spring stiffness Kt"""
    def __init__(self, gap, w, **kwargs):
        self.gap = gap
        self.w = np.atleast_1d(w)
        self.n_ny = 1
        super().__init__(**kwargs)
        
    def set_active(self,n,m,p,q):
        # all are active
        self._active = np.s_[0:q*self.n_nl]
        
    def fnl(self, x,y,u):
        y = np.atleast_2d(y)
        ynl =np.inner(self.w, y)  # (n_nx, ns)
        return (ynl-self.gap) * (ynl - self.gap >= 0).astype(float)

    def dfdx(self,x,y,u):
        return np.array([])

    def dfdy(self,x,y,u):
        w = self.w
        y = np.atleast_2d(y)
        ynl = np.inner(w, y)
        dfdy = np.einsum('i,j,k->ikj',w, 
                        (ynl - self.gap >= 0).astype(float), w)
        return dfdy    

class Tanhdryfriction(Nonlinear_Element):
    """Friction model. sign(ẏ) approximated by tanh. eps control the slope.
    Make sure the velocity is included in the output of the state space model
    """
    def __init__(self, eps, w, **kwargs):
        self.eps = eps
        self.w = np.atleast_1d(w)
        self.n_ny = 1
        super().__init__(**kwargs)

    def set_active(self,n,m,p,q):
        # all are active
        self._active = np.s_[0:q*self.n_nl]
    
    def fnl(self, x,y,u):
        y = np.atleast_2d(y)
        # displacement of dofs attached to nl
        ynl =np.inner(self.w, y)  # (n_nx, ns)
        f = np.tanh(ynl / self.eps)
        return f

    def dfdx(self,x,y,u):
        return np.array([])

    def dfdy(self,x,y,u):
        w = self.w
        y = np.atleast_2d(y)
        ynl = np.inner(w, y)
        dfdy = np.einsum('i,j,k->ikj',w, 
                         (1 - np.tanh(ynl / self.eps)**2) / self.eps, w)
        return dfdy

class Pnlss(Nonlinear_Element):
    """Combinations of monomials in x and u"""

    def __init__(self, degree, structure, eq = None, **kwargs):
        """Initialize active nonlinear terms/monomials to be optimized"""
        self.degree = np.asarray(degree)
        self.structure = structure
        self.eq = eq
        super().__init__(**kwargs)
    
    def set_active(self,n,m,p,q):
        # all possible terms
        self.powers = combinations(n+m, self.degree)
        self.n_nx = self.powers.shape[0]
        
        # for backward-compability
        if self.eq in ('state', 'x'): q = n
        if self.eq in ('output', 'y'): q = p
        # q = n for 'x' and q = p for 'y'
        active = select_active(self.structure,n,m,q,self.degree)
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
        # number of x signals wrt which derivatives are taken
        n = np.atleast_2d(x).shape[1]
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


class Polynomial_x(Nonlinear_Element):
    def __init__(self, exponent, w, **kwargs):
        self.w = np.atleast_2d(w)
        self.exponent = np.atleast_1d(exponent)

        super().__init__(**kwargs)

        # number of nonlinear elements
        self.n_nx = 1
        self.n_ny = 0
        
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
    
    Example
    -------
    fnl = (y₁-y₂)ẏ₁, where y = [y₁, y₂, ẏ₁, ẏ₂]
    exponents = [1,1]
    w = np.array([[1,-1,0,0], [0,1,0,0]]
    """

    def __init__(self, exponent, w, structure='Full',**kwargs):
        """
        exponents: ndarray (n_ny)
        w: ndarray (n_ny, p)
        """
        self.w = np.atleast_2d(w)
        self.exponent = np.atleast_1d(exponent)
        self.structure = structure
        super().__init__(**kwargs)
        # number of nonlinear elements
        self.n_nx = 0
        self.n_ny = 1
        
    def set_active(self,n,m,p,q):
        # all are active
        self._active = np.r_[0:q*self.n_nl]

    def fnl(self, x,y,u):
        y = np.atleast_2d(y)
        # displacement of dofs attached to nl
        ynl = y @ self.w.T  # [nt, yactive]
        fnl = np.prod(ynl**self.exponent, axis=1)
        return fnl

    def dfdx(self,x,y,u):
        """Output nl, thus zero"""
        return np.array([])

    def dfdy(self,x,y,u):
        """
        dfdy (1, p, ns)
        """
        exp = self.exponent
        y = np.atleast_2d(y)
        ynl = y @ self.w.T

        Ptmp = np.eye(self.w.shape[0])
        dfdyP = np.zeros((1, len(exp), y.shape[0]))  # [self.n_ny, yactive,nt]
        # derivative wrt. each coloumn in ynl(or row in w). Denoted yactive
        for ip in range(self.w.shape[0]):
            dfdyP[0,ip,:] = exp[ip]*np.prod(ynl**(exp-Ptmp[ip]), axis=1)

        # derivative wrt all y.
        dfdy = np.einsum('ijk,jl->ilk',dfdyP, self.w)
        
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
            nl.set_active(m,n,p,q)
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
        self.xactive = np.array([],dtype=np.intp)
        self.yactive = np.array([],dtype=np.intp)
        n_nl = 0
        for nl in self.nls:
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
        # TODO can only use either x or y elements, note the bool! hack
        nlj = 0
        self.jac_x = np.array([],dtype=np.intp)
        self.jac_y = np.array([],dtype=np.intp)
        for nl in self.nls:
            active = np.r_[nl.active]
            self.jac_x = np.r_[self.jac_x, nlj                       +\
                          np.r_[:bool(nl.n_nx)*len(active)]]
            self.jac_y = np.r_[self.jac_y, nlj + nl.n_nx*len(active) +\
                          np.r_[:bool(nl.n_ny)*len(active)]]
            nlj += nl.n_nl*len(active)

        n_nx = 0
        for nl in self.nls:
            # convert local index to global index. Active elements in global E
            npar = nl.n_nx
            if npar == 0:
                continue
            active = np.r_[nl.active]  # active might be a slice -> convert
            col = np.mod(active, npar)
            row = (active-col)// npar
            idx = row*self.n_nx + col
            self.xactive = np.r_[self.xactive, n_nx + idx]
            n_nx += npar
        
        n_ny = 0
        for nl in self.nls:
            # convert local index to global index. Active elements in global E
            npar = nl.n_ny
            if npar == 0:
                continue
            active = np.r_[nl.active]  # active might be a slice -> convert
            col = np.mod(active, npar)
            row = (active-col)// npar
            idx = row*self.n_ny + col
            self.yactive = np.r_[self.yactive, n_ny + idx]
            n_ny += npar


    def fnl(self,x,y,u):
        """
        x: ndarray (n, ns)
        y: ndarray (p, ns)
        u: ndarray (m, ns)

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



#t = np.arange(100)/100
#u  = np.atleast_2d(np.sin(2*np.pi*t)).T
#y = np.empty((100,2))
#y[:,0] = np.sin(2*np.pi*t)
#y[:,1] = np.sin(np.pi*t)
#eps = 0.05
#tanh1 = Tanhdryfriction(eps=eps,w=[1,0])
#tanh1.set_active(2,1,1,2)
#fnl = tanh1.fnl(0,y,0)
#dfdy = tanh1.dfdy(0,y,0)

#poly1y = Polynomial(exponent=2,w=1)
#poly2y = Polynomial(exponent=3,w=1)
#poly3y = Polynomial(exponent=4,w=1)
#
#poly1x = Polynomial_x(exponent=2,w=[0,1])
#poly2x = Polynomial_x(exponent=3,w=[0,1])
#poly3x = Polynomial_x(exponent=4,w=[0,1])
## nlx2 = NLS([poly1,poly2])  #,poly3])
##nlx2 = NLS([poly2x,poly1y,poly3y])  #,poly3])
#nlx2 = NLS([poly1y,poly3y,poly2x,poly2y])  #,poly3])
#nly2 = NLS([poly1x,poly2x])
##nlx2 = NLS([poly2x, poly1y])
#
##nlx2 = NLS([Pnlss('x', 2, 'full')])
#nlx1 = NLS([Pnlss(degree=[2], structure='full')])
#
#nlx1.set_active(2,1,1,2)
#nlx2.set_active(2,1,1,2)
#
#nly2.set_active(2,1,1,1)

#from pyvib.polynomial import multEdwdx, nl_terms
#
#pnlss = Pnlss(degree=2, structure='full',eq='x')    
#pnlss.set_active(2,1,1,2)
#
#t = np.arange(100)/100
#u  = np.atleast_2d(np.sin(2*np.pi*t)).T
#x = np.random.rand(100,2)
#
##x = np.array([2,1])
##u = np.array([1])
#signal = np.hstack((x, u)).T
#zeta = nl_terms(signal, pnlss.powers)  # (n_nx, nts)
#fnl = pnlss.fnl(x,0,u)
#print(np.allclose(zeta, fnl))
#
#dhdy = pnlss.dfdx(x,1,u)
#signal = np.atleast_2d(np.hstack((x, u))).T
#n = 2
#E = np.arange(n*pnlss.n_nx).reshape(n,pnlss.n_nx)
#
#Edx = multEdwdx(signal, pnlss.d_powers, pnlss.d_coeff, E, n)
#Gdidy = np.einsum('ij,jkl->ikl',E,dhdy)
#print(np.allclose(Edx, Gdidy))

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
