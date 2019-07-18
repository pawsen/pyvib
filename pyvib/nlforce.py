#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import coo_matrix
from .interpolate import spline, piecewise_linear, piecewise_linear_der

class NL_force(object):

    def __init__(self, nls=None):
        self.nls = []
        self.dnls_force = []
        self.dnls_damp = []

        if nls is not None:
            self.add(nls)

    def add(self, nls):
            if not isinstance(nls, list):
                nls = [nls]
            for nl in nls:
                self.nls.append(nl)
                if nl.is_force:
                    self.dnls_force.append(nl)
                else:
                    self.dnls_damp.append(nl)

    def nldofs(self):
        # find nonlinear dofs
        nldofs = []
        for nl in self.nls:
            inl = nl.inl
            for connection in inl:
                i1 = connection[0]
                i2 = connection[1]
                if i1 != -1:
                    nldofs.append(i1)
                if i2 != -1:
                    nldofs.append(i1)
        # Get unique elements (could also be np.unique(np.asarray(nldofs)))
        nldofs = np.asarray(list(set(nldofs)))
        return nldofs


    def force(self, x, xd):

        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        fnl = np.zeros((ndof+1, ns))

        for nl in self.nls:
            fnl = nl.compute(x, xd, fnl)
        # squeeze in case ns = 1
        fnl = fnl[:ndof,:].squeeze()
        return fnl

    def dforce(self, x, xd, is_force=True):
        """Derivative of nonlinear functional
        """
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape

        dfnl = np.zeros((ndof+1, ns*(ndof+1)))

        if is_force:
            for nl in self.dnls_force:
                dfnl = nl.dcompute(x, xd, dfnl)
        else:
            for nl in self.dnls_damp:
                dfnl = nl.dcompute(x, xd, dfnl)

        if ns == 1:
            return dfnl[:ndof,:ndof].squeeze()

        # create sparse structure from dfnl
        # TODO: dont create dfnl in the first place...:)
        ind = np.arange(ns*(ndof+1))
        ind = np.delete(ind, np.s_[ndof::ndof+1])
        dfnl = dfnl[:ndof, ind]
        dfnl = np.reshape(dfnl, (ndof**2,ns), order='F')
        # dont ask...
        ind = np.outer(np.ones(ndof), np.arange(ndof)) * ns*ndof + \
            np.outer(np.arange(ndof), np.ones(ndof))
        ind = np.outer(ind.T, np.ones(ns)) + \
            ns*ndof * np.outer(np.ones(ndof**2), np.arange(0,(ns-1)*ndof+1, ndof)) + \
            ndof * np.outer(np.ones(ndof**2), np.arange(ns))

        ind = ind.ravel(order='F').astype('int')

        #https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python
        arr = ns*ndof*np.array([1,1])
        ii, jj = np.unravel_index(ind, tuple(arr), order='F')
        dfnl_s = coo_matrix((dfnl.ravel(order='F'), (ii, jj)),
                            shape=(ndof*ns, ndof*ns)).tocsr()

        return dfnl_s
        # return dfnl

    def energy(self, x, xd):
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        energy = 0
        for nl in self.nls:
            energy = nl.energy(x, xd, energy)
        return energy


class _NL_compute(object):
    is_force = True

    def compute(self, x, fnl):
        pass
    def dcompute(self, x, fnl):
        pass
    def energy(self, x, fnl):
        pass

class NL_polynomial(_NL_compute):
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

    def compute(self, x, xd, fnl):
        inl = self.inl
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            # connected from
            i1 = inl[j,0]
            # conencted to
            i2 = inl[j,1]

            # Convert to the right index
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]

            # if connected to ground
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            # in case of even functional
            if (self.enl[j] % 2 == 0):
                x12 = np.abs(x12)

            f12 = self.knl[j] * x12**self.enl[j]
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        """Derivative of nonlinear functional
        """
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = (np.array([ndof]),)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            df12 = self.knl[j] * self.enl[j] * np.abs(x12)**(self.enl[j]-1)
            # in case of even functional
            if (self.enl[j] % 2 == 0):
                idx = np.where(x12 < 0)
                df12[idx] = -df12[idx]

            # extraction needed due to the way the slice is made in dfnl.
            id1 = idx1[0][0]
            id2 = idx2[0][0]

            # add the nonlinear force to the right dofs
            dfnl[idx1, id1::ndof+1] += df12
            dfnl[idx2, id1::ndof+1] -= df12
            dfnl[idx1, id2::ndof+1] -= df12
            dfnl[idx2, id2::ndof+1] += df12
        return dfnl

    def energy(self, x, xd, energy):
        inl = self.inl
        if self.is_force is False:
            x = xd
        if x.ndim == 1:
            ndof, = x.shape
            ns = 1
        else:
            ndof, ns = x.shape
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if (self.enl[j] % 2 == 0):
                x12 = np.abs(x12)
            e12 = self.knl[j] / (self.enl[j]+1) * abs(x12)**(self.enl[j]+1)
            # TODO there should be more to this if-statement
            if x12 < 0:
                pass
                #e12 = 0
            energy += e12
        return energy

class NL_tanh_damping(_NL_compute):
    def __init__(self, inl, enl, knl):
        self.inl = inl
        self.enl = enl
        self.knl = knl
        self.is_force = False

    def compute(self, x, xd, fnl):
        inl = self.inl
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            xd1 = xd[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                xd2 = 0
            else:
                idx2 = np.where(i2 == idof)
                xd2 = xd[idx2]
            xd12 = xd1 - xd2
            f12 = self.knl[j] * np.tanh(xd12 * self.enl[j])
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            xd1 = xd[idx1]
            if i2 == -1:
                idx2 = (np.array([ndof]),)
                xd2 = 0
            else:
                idx2 = np.where(i2 == idof)
                xd2 = xd[idx2]
            xd12 = xd1 - xd2
            df12 = self.enl[j] * self.knl[j] * (1 -
                                                np.tanh(xd12 * self.enl[j])**2)

            id1 = idx1[0][0]
            id2 = idx2[0][0]
            dfnl[idx1, id1::ndof+1] += df12
            dfnl[idx2, id1::ndof+1] -= df12
            dfnl[idx1, id2::ndof+1] -= df12
            dfnl[idx2, id2::ndof+1] += df12
        return dfnl

class NL_piecewise_linear(_NL_compute):
    def __init__(self, x, y, slope, inl, delta=None, symmetric=False,
                 is_force=True):
        """
        Parameters
        ----------
        x: ndarray [nbln, n_knots]
            x-coordinates for knots
        y: ndarray [nbln, n_knots]
            y-coordinates for knots
        slope: ndarray [nbln, n_knots+1]
            Slope for each linear segment
        delta: ndarray [nbln, n_knots]
            Regularization length, ie. enforce continuity of the derivative.
        inl: ndarray [nbln, 2]
            DOFs for nonlinear connection
        """
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.slope = np.asarray(slope)
        self.delta = delta
        self.inl = inl
        self.is_force = is_force
        if not isinstance(symmetric,(list)) and inl.shape[0] > 1:
             symmetric = [symmetric]*inl.shape[0]
        self.symmetric = symmetric

    def compute(self, x, xd, fnl):
        inl = self.inl
        nbln = inl.shape[0]
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric is True and x12 < 0:
                x12 = -x12
            f12 = piecewise_linear(self.x, self.y, self.slope,
                                   self.delta, x12)
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = (np.array([ndof]),)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric is True and x12 < 0:
                x12 = -x12
            df12 = piecewise_linear_der(self.x, self.y, self.slope,
                                        self.delta, x12)

            id1 = idx1[0][0]
            id2 = idx2[0][0]
            dfnl[idx1, id1::ndof+1] += df12
            dfnl[idx2, id1::ndof+1] -= df12
            dfnl[idx1, id2::ndof+1] -= df12
            dfnl[idx2, id2::ndof+1] += df12

        return dfnl

class NL_spline(_NL_compute):
    def __init__(self, x, coeff, symmetric, inl, is_force=True):
        self.x = x
        self.coeff = coeff
        self.symmetric = symmetric
        self.inl = inl
        self.is_force = is_force

    def compute(self, x, xd, fnl):
        inl = self.inl
        nbln = inl.shape[0]
        ndof = fnl.shape[0] - 1
        idof = np.arange(ndof)
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = ([ndof],)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            f12, _ = spline(self.x[j], self.coeff[j], x12)
            fnl[idx1] += f12
            fnl[idx2] -= f12
        return fnl

    def dcompute(self, x, xd, dfnl):
        inl = self.inl
        ndof = dfnl.shape[0]-1
        idof = np.arange(ndof)
        nbln = inl.shape[0]
        if self.is_force is False:
            x = xd

        for j in range(nbln):
            i1 = inl[j,0]
            i2 = inl[j,1]
            idx1 = np.where(i1 == idof)
            x1 = x[idx1]
            if i2 == -1:
                idx2 = (np.array([ndof]),)
                x2 = 0
            else:
                idx2 = np.where(i2 == idof)
                x2 = x[idx2]
            x12 = x1 - x2
            if self.symmetric[j] is True and x12 < 0:
                x12 = -x12
            _, df12 = spline(self.x[j], self.coeff[j], x12)

            id1 = idx1[0][0]
            id2 = idx2[0][0]
            dfnl[idx1, id1::ndof+1] += df12
            dfnl[idx2, id1::ndof+1] -= df12
            dfnl[idx1, id2::ndof+1] -= df12
            dfnl[idx2, id2::ndof+1] += df12
        return dfnl
