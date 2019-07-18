#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy.linalg import norm, solve

class Newmark():
    def __init__(self, M, C, K, nonlin, gtype=None):
        self.M = M
        self.C = C
        self.K = K
        self.nonlin = nonlin
        if gtype is None:
            self.gamma = 1/2
            self.beta = 1/4
        else:
            self.gamma, self.beta = self._params(gtype)

    def integrate_nl(self, x0, xd0, dt, fext, sensitivity=False):
        return newmark_beta_nl(self.M, self.C, self.K, x0, xd0, dt, fext,
                               self.nonlin, sensitivity, self.gamma, self.beta)

    def _params(self, gtype):
        # (gamma, beta)
        d = {
            'explicit':    (0,   0),
            'central':     (1/2, 0),
            'fox-goodwin': (1/2, 1/12),
            'linear':      (1/2, 1/6),
            'average':     (1/2, 1/4)
            }
        try:
            gamma, beta = d[gtype]
        except KeyError as err:
            raise Exception('Wrong key {}. Should be one of {}'.
                            format(gtype, d.keys())) from err

        return gamma, beta

def newmark_beta_nl(M, C, K, x0, xd0, dt, fext, nonlin, sensitivity=False,
                    gamma=1/2, beta=1/4):
    '''Newmark-beta nonlinear integration.

    With gamma = 1/2, beta = 1/4, this correspond to the "Average
    acceleration" Method. Unconditional stable. Convergence: O(dt**2).

    No enforcing of boundary conditions, eg. only solves IVP.
    Input:
        xo, xd0
        - Initial conditions. Size [ndof]
        t
        - Time vector. Size[nsteps]
        r_ext(t)
        - External force function.
        - Takes the current time as input.
        - Returns an array. Size [ndof]

    Output:
        x, xd, xdd
        - State arrays. Size [nsteps, ndof]

    Equations are from Krenk: "Non-linear Modeling and analysis of Solids
        and Structures"
    See also: Cook: "Concepts and applications of FEA, chap 11 & 17"

    '''
    tol = 1e-10
    itmax = 50

    A1 = (1 - gamma) * dt
    B1 = (1/2 - beta) * dt**2
    A2 = 1 / beta / dt**2
    B2 = gamma / beta / dt

    fext = fext.T
    ns = fext.shape[0]
    ndof = K.shape[0]
    # Pre-allocate arrays.
    xdd = np.empty((ns, ndof))
    xd = np.empty((ns, ndof))
    x = np.empty((ns, ndof))
    S_lin = K + gamma / beta / dt * C + M / beta / dt**2

    x[0] = x0
    xd[0] = xd0
    # initial acceleration. eq. 11.12-13
    fl = C @ xd[0] + K @ x[0]
    fnl = nonlin.force(x[0], xd[0])
    xdd[0] = solve(M, fext[0] - fl - fnl)

    if sensitivity:
        V = np.hstack((np.eye(ndof), np.zeros((ndof, ndof))))
        dV = np.hstack((np.zeros((ndof, ndof)), np.eye(ndof)))
        dfnl = nonlin.dforce(x[0], xd[0], is_force=True)
        dfnl_d = nonlin.dforce(x[0], xd[0], is_force=False)
        rhs = -(C + dfnl_d) @ dV - (K + dfnl) @ V
        ddV = solve(M, rhs)

    # time stepping
    for j in range(1, ns):
        #dt = t[j] - t[j-1]
        # Prediction step
        xd[j] = xd[j-1] + A1 * xdd[j-1]
        x[j] = x[j-1] + dt * xd[j-1] + B1 * xdd[j-1]
        xdd[j] = 0  # xdd[j-1]

        # force at current step
        fl = C @ xd[j] + K @ x[j]
        fnl = nonlin.force(x[j], xd[j])
        res = M @ xdd[j] + fl + fnl - fext[j]

        it = 0
        dx = 1
        # correct prediction step
        # TODO break-criterion: norm(res)> eps*norm(f_glob)
        while(norm(res) > tol and it < itmax):
            # system matrices and increment correction
            """ calculate tangent stiffness.
            r(u) : residual
            Kt   : Tangent stiffness. Kt = ∂r/∂u

            r(u,u̇) = Cu̇ + Ku + f(u,u̇) - p
            Kt = ∇{u}r = K + ∇{u}f
            Ct = ∇{u̇}r = C + ∇{u̇}f
            """

            dfnl = nonlin.dforce(x[j], xd[j], is_force=True)
            dfnl_d = nonlin.dforce(x[j], xd[j], is_force=False)

            Seff = dfnl + dfnl_d * B2 + S_lin
            dx = - solve(Seff, res)
            xdd[j] += A2 * dx
            xd[j] += B2 * dx
            x[j] += dx

            fl = C @ xd[j] + K @ x[j]
            fnl = nonlin.force(x[j], xd[j])
            res = M @ xdd[j] + fl + fnl - fext[j]

            it += 1
            #print("j: {}, i: {}, delta_x: {}, res: {}, xd_norm: {}".
            #      format(j,i,delta_x,res_norm,delta_x_norm))

        if it == itmax:
            raise ValueError('Max iteration reached')
        if sensitivity:
            dfnl = nonlin.dforce(x[j], xd[j], is_force=True)
            dfnl_d = nonlin.dforce(x[j], xd[j], is_force=False)
            V = V + dt*dV + (1/2 - beta) * dt**2 * ddV
            dV = dV + (1 - gamma) * dt * ddV
            S = dfnl + S_lin + gamma/beta/dt * dfnl_d
            S = dt**2 * beta * S
            rhs = -(C + dfnl_d) @ dV - (K + dfnl) @ V
            ddV = solve(S, rhs)
            V = V + dt**2 * beta * ddV
            dV = dV + dt * gamma * ddV
    if sensitivity:
        return x.T, xd.T, xdd.T, np.vstack((V, dV))
    else:
        return x.T, xd.T, xdd.T


def newmark_beta_lin(M, C, K, x0, xd0, t, r_ext):
    '''
    Newmark-beta linear integration.
    With gamma = 1/2, beta = 1/4, this correspond to the "Average acceleration"
    Method. Unconditional stable. Convergence: O(dt**2).

    No enforcing of boundary conditions, eg. only solves IVP.
    Input:
        M, C, K
        - System matrices. Size [ndof, ndof]
        xo, xd0
        - Initial conditions. Size [ndof]
        t
        - Time vector. Size[nsteps]

    Output:
        x, xd, xdd
        - State arrays. Size [nsteps, ndof]

    Equations are from Cook: "Concepts and applications of FEA"
    '''
    gamma = 1./2
    beta = 1./4

    nsteps = len(t)
    ndof = M.shape[0]
    # Pre-allocate arrays
    xdd = np.zeros([nsteps, ndof], dtype=float)
    xd = np.zeros([nsteps, ndof], dtype=float)
    x = np.zeros([nsteps, ndof], dtype=float)

    x[0,:] = x0
    xd[0,:] = xd0
    # initial acceleration. eq. 11.12-13
    r_int = np.dot(K,x[0,:]) + np.dot(C,xd[0,:])
    xdd[0,:] = linalg.solve(M, r_ext(0) - r_int)

    # time stepping
    for j in range(1, nsteps):
        dt = t[j] - t[j-1]
        a0 = 1./(beta * dt**2)
        a1 = 1./(beta * dt)
        a2 = (1./(2*beta) - 1)
        a3 = gamma/(beta * dt)
        a4 = (gamma/beta - 1)
        a5 = dt*(gamma/(2*beta) - 1)

        # 11.13-5b
        Keff = a0 * M + a3 * C + K
        # 11.13-5a
        M_int = np.dot(M, a0 * x[j-1,:] + a1 * xd[j-1,:] + a2 * xdd[j-1,:])
        C_int = np.dot(C, a3 * x[j-1,:] + a4 * xd[j-1,:] + a5 * xdd[j-1,:])
        x[j,:] = linalg.solve(Keff, M_int + C_int + r_ext(t[j]))

        # update vel and accel, eq. 11.13-4a & 11.13-4b
        xdd[j,:] = a0 * (x[j,:] - x[j-1,:] - dt*xd[j-1,:]) - a2 * xdd[j-1,:]
        xd[j,:] = a3 * (x[j,:] - x[j-1,:]) - a4 * xd[j-1,:] - dt * a5 * xdd[j-1,:]
    return x, xd, xdd
