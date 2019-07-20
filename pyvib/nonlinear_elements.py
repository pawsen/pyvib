#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Nonlinear_Element:
    """Bare class
    """
    pass

class Polynomial_x(Nonlinear_Element):
    def __init__(self, exponent, w):
        self.w = np.atleast_2d(w)
        self.exponent = np.atleast_1d(exponent)
        # number of nonlinear elements
        self.n_nx = 1
        self.n_ny = 0
        self.n_nl = self.n_nx + self.n_ny
        self._active = np.array([],dtype=np.intp)

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
    
    @property
    def active(self):
        return self._active
        
    def set_active(self,n,m,p,q):
        # all are active
        self._active = np.s_[0:q*self.n_nl]



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
            # Find columns in E(n,n_nl) matrix which correspond to x-dependent
            # nl (idx) and y-dependent (idy). Note it does not matter if we 
            # have '+nl.n_nx' in self.idy or '+nl.n_ny' in self.idx
            self.idx = np.r_[self.idx, self.n_nl         + np.r_[0:nl.n_nx]]
            self.idy = np.r_[self.idy, self.n_nl+nl.n_nx + np.r_[0:nl.n_ny]]

            self.nls.append(nl)
            self.n_nx += int(nl.n_nx)
            self.n_ny += int(nl.n_ny)
            self.n_nl += int(nl.n_nl)
            
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
        
        n_nl = 0
        # If the same NLS object is used multiple times, we need to
        # reset the active count, as done here.
        self.active = np.array([],dtype=np.intp)
        self.jac_active = np.array([],dtype=np.intp)
        self.xactive = np.array([],dtype=np.intp)
        self.yactive = np.array([],dtype=np.intp)
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
        
        for i, nl in enumerate(self.nls):
            fnl[i] = nl.fnl(x,y,u)

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
        i = 0
        for nl in self.nls:
            tmp = nl.dfdy(x,y,u)
            if tmp.size:
                dfdy[i] = tmp
                i += 1
        
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
        i = 0
        for nl in self.nls:
            tmp = nl.dfdx(x,y,u)
            if tmp.size:
                dfdx[i] = tmp
                i += 1
        
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
