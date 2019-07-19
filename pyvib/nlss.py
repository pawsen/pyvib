#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d
from pyvib.nonlinear_elements import Polynomial, NLS, Polynomial_x
from pyvib.statespace import NonlinearStateSpace, StateSpaceIdent

from scipy import signal
from scipy.linalg import solve


class NLSS(NonlinearStateSpace, StateSpaceIdent):
    def __init__(self, nlx, nly, *system, **kwargs):
        """
        """
        if len(system) == 1:  # and isinstance(system[0], StateSpace):
            sys = system
            kwargs['dt'] = sys[0].dt
        else:  # given as A,B,C,D
            sys = system
            kwargs['dt'] = 1  # unit sampling

        super().__init__(*sys, **kwargs)
        self.nlx = nlx
        self.nly = nly
        self.E = np.zeros((self.n, self.nlx.n_nl))
        self.F = np.zeros((self.p, self.nly.n_nl))
        
        # set active elements(if needed) now the system size is known
        nlx.set_active(self.n,self.m,self.p,self.n)
        nly.set_active(self.n,self.m,self.p,self.p)
        
    def output(self, u, t=None, x0=None):
        return dnlsim(self, u, t=t, x0=x0)
    
    def jacobian(self, x0, weight=False):
        return jacobian(x0, self, weight=weight)
        
def dnlsim(system, u, t=None, x0=None):
    """Simulate output of a discrete-time nonlinear system.

    Calculate the output and the states of a nonlinear state-space model.
        x(t+1) = A x(t) + B u(t) + E h(x(t),u(t)) + G i(y(t),ẏ(t))
        y(t)   = C x(t) + D u(t) + F j(x(t),u(t))

    The initial state is given in x0.

    """
    #if not isinstance(system, NLSS):
    #    raise ValueError(f'System must be a NLSS object {type(system)}')

    u = np.atleast_1d(u)
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

    # Simulate the system
    # the output from fnl is (n_nl,1). The time stepping expect (n_nl)
    for i in range(0, out_samples - 1):
        # Output equation y(t) = C*x(t) + D*u(t) + F*j(t)
        # TODO jvec: must not depend on y!
        jvec = system.nly.fnl(xout[i],0,u_dt[i]).squeeze()
        yout[i, :] = (np.dot(system.C, xout[i, :]) +
                      np.dot(system.D, u_dt[i, :]) + 
                      np.dot(system.F, jvec))
        # State equation x(t+1) = A*x(t) + B*u(t) + E*zeta(y(t),ẏ(t))
        hvec = system.nlx.fnl(xout[i, :],yout[i, :],u_dt[i, :]).squeeze()
        xout[i+1, :] = (np.dot(system.A, xout[i, :]) +
                        np.dot(system.B, u_dt[i, :]) +
                        np.dot(system.E, hvec))

    # Last point
    jvec = system.nly.fnl(xout[-1, :],0,u_dt[-1, :]).squeeze()
    yout[-1, :] = (np.dot(system.C, xout[-1, :]) +
                   np.dot(system.D, u_dt[-1, :]) +
                   np.dot(system.F, jvec))

    return tout, yout, xout

def jacobian(x0, system, weight=False):
    """Compute the Jacobians of a steady state nonlinear state-space model

    Jacobians of a nonlinear state-space model

        x(t+1) = A x(t) + B u(t) + E h(x(t),u(t)) + G i(y(t),ẏ(t))
        y(t)   = C x(t) + D u(t) + F j(x(t),u(t))

    i.e. the partial derivatives of the modeled output w.r.t. the active
    elements in the A, B, E, F, D, and C matrices, fx: JA = ∂y/∂Aᵢⱼ

    x0 : ndarray
        flattened array of state space matrices

    """

    n, m, p = system.n, system.m, system.p
    #R, npp = system.signal.R, system.signal.npp
    R, npp = system.R, system.npp

    # total number of points
    ns = R*npp
    # TODO without_T2 = system.without_T2

    # Collect states and outputs with prepended transient sample
    y_trans = system.y_mod[system.idx_trans]
    x_trans = system.x_mod[system.idx_trans]
    #u_trans = system.signal.um[system.idx_trans]
    u_trans = system.um[system.idx_trans]
    nts = u_trans.shape[0]  # nts: number of total samples(including transient)
    
    A, B, C, D, E, F = system.extract(x0)
    # split E in x- and y dependent part
    G = E[:,system.nlx.idy]
    E = E[:,system.nlx.idx]
    
    fnl = system.nlx.fnl(x_trans,y_trans,u_trans)  # (n_nx, nts)
    hvec = fnl[system.nlx.idx].T  # (nts, nlx.n_nx)
    ivec = fnl[system.nlx.idy].T  # (nts, nlx.n_ny)
    jvec = system.nly.fnl(x_trans,y_trans,u_trans).T  # (nts, n_ny)
    
    # derivatives of nonlinear functions wrt x or y
    # (n_nx,n,ns)
    dhdx = system.nlx.dfdx(x_trans,y_trans,u_trans)
    didy = system.nlx.dfdy(x_trans,y_trans,u_trans)
    djdx = system.nly.dfdy(x_trans,y_trans,u_trans)
    
    if E.size == 0:
        A_Edhdx = np.zeros(shape=(*A.shape,nts))
    else:
        A_Edhdx = np.einsum('ij,jkl->ikl',E,dhdx)  # (n,n,nts)
    if G.size == 0:
        Gdidy = np.zeros(shape=(*A.shape,nts))
    else:
        Gdidy = np.einsum('ij,jkl->ikl',G,didy)  # (n,p,nts)
    if F.size == 0:
        C_Fdjdx = np.zeros(shape=(*C.shape,nts))
    else:
        C_Fdjdx = np.einsum('ij,jkl->ikl',F,djdx)  # (p,n,nts)
    A_Edhdx += A[...,None]
    C_Fdjdx += C[...,None]
    
    # calculate output jacobians wrt state space matrices in output eq
    JC = np.kron(np.eye(p), system.x_mod)  # (p*ns,p*n)
    #JD = np.kron(np.eye(p), system.signal.um)  # (p*ns, p*m)
    JD = np.kron(np.eye(p), system.um)  # (p*ns, p*m)
    if system.nly.yactive.size:
        JF = np.kron(np.eye(p), jvec)  # Jacobian wrt all elements in F
        JF = JF[:,system.nly.yactive]  # all active elements in F. (p*nts,nactiveF)
        JF = JF[system.idx_remtrans]  # (p*ns,nactiveF)
    else:
        JF = np.array([]).reshape(p*ns,0)
    
    # calculate Jacobian by filtering an alternative state-space model
    # reshape so first row of JA is the derivative wrt all elements in A for
    # first time step, first putput, then second output, then next time,...
    JA = element_jacobian(x_trans, A_Edhdx, Gdidy, C_Fdjdx, np.arange(n**2))
    JA = JA.transpose((2,0,1)).reshape((nts*p, n**2))
    JA = JA[system.idx_remtrans]  # (p*ns,n**2)

    JB = element_jacobian(u_trans, A_Edhdx, Gdidy, C_Fdjdx, np.arange(n*m))
    JB = JB.transpose((2,0,1)).reshape((nts*p, n*m))
    JB = JB[system.idx_remtrans]  # (p*ns,n*m)

    if system.nlx.xactive.size:
        JE = element_jacobian(hvec, A_Edhdx, Gdidy, C_Fdjdx, system.nlx.xactive)
        JE = JE.transpose((2,0,1)).reshape((nts*p, len(system.nlx.xactive)))
        JE = JE[system.idx_remtrans]  # (p*ns,nactiveE)
    else:
        JE = np.array([]).reshape(p*ns,0)

    if system.nlx.yactive.size:
        JG = element_jacobian(ivec, A_Edhdx, Gdidy, C_Fdjdx, system.nlx.yactive)
        JG = JG.transpose((2,0,1)).reshape((nts*p, len(system.nlx.yactive)))
        JG = JG[system.idx_remtrans]  # (p*ns,nactiveE)
    else:
        JG = np.array([]).reshape(p*ns,0)

    
    jac = np.hstack((JA, JB, JC, JD, JE, JG, JF))  # TODO [without_T2]
    jac = np.hstack((JA, JA))
    npar = jac.shape[1]
    
    return jac


def element_jacobian(samples, A_Edhdx, Gdidy, C_Fdjdx, active):
    """Compute Jacobian of the output y wrt. A, B, and E

    The Jacobian is calculated by filtering an alternative state-space model


    ∂x∂Aᵢⱼ(t+1) = Iᵢⱼx(t) + (A + E*∂h∂x) ∂x∂Aᵢⱼ(t) + G*∂i∂y*∂y∂Aᵢⱼ(t)
    ∂y∂Aᵢⱼ(t) = (C + F)*∂x∂Aᵢⱼ(t)

    ∂x∂Bᵢⱼ(t+1) = Iᵢⱼu(t) + (A + E*∂h∂x) ∂x∂Aᵢⱼ(t) + G ∂i∂y*∂y∂Aᵢⱼ(t)
    ∂y∂Bᵢⱼ(t) = (C + F)*∂x∂Bᵢⱼ(t)

    ∂x∂Eᵢⱼ(t+1) = Iᵢⱼ(t) + (A + E*∂h∂x) ∂x∂Aᵢⱼ(t) + G ∂i∂y*∂y∂Aᵢⱼ(t)
    ∂y∂Eᵢⱼ(t) = (C + F)*∂x∂Eᵢⱼ(t)

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
    out : ndarray (p,nactive,nt)
       Jacobian; JA, JB or JE depending on the samples given as input

    See fJNL

    """
    # Number of outputs and number of states
    n, n, nt = A_Edhdx.shape
    p, n, nt = C_Fdjdx.shape
    # number of inputs and number of samples in alternative state-space model
    nt, npar = samples.shape
    nactive = len(active)  # Number of active parameters in A, B, or E

    out = np.zeros((p,nactive,nt))
    for k, activ in enumerate(active):
        # Which column in A, B, or E matrix
        j = np.mod(activ, npar)
        # Which row in A, B, or E matrix
        i = (activ-j)//npar
        # partial derivative of x(0) wrt. A(i,j), B(i,j), or E(i,j)
        J = np.zeros(n)
        for t in range(0,nt-1):
            # Calculate output alternative state-space model at time t
            out[:,k,t] = C_Fdjdx[:,:,t] @ J
            J = A_Edhdx[:,:,t] @ J + Gdidy[:,:,t] @ out[:,k,t]
            J[i] += samples[t,j]
        # last time point
        out[:,k,-1] = C_Fdjdx[:,:,t] @ J

    return out


# test linear system
m = 1
k = 2
d = 3
fex = 1
dt = 0.1

# cont. time formulation
A = np.array([[0, 1],[-k/m, -d/m]])
B = np.array([[0], [fex/m]])
C = np.array([[1, 0]])
D = np.array([[0]])
sys = signal.StateSpace(A, B, C, D).to_discrete(dt)

# add polynomial in state eq
exponent = [3]
w = [1]  # need to be same length as number of ouputs
poly1 = Polynomial(exponent,w)
poly2 = Polynomial_x(exponent,w=[0,1])
poly3 = Polynomial_x(exponent=[2],w=[0,1])
poly4 = Polynomial(exponent=[5],w=[-1])
poly5 = Polynomial(exponent=[2],w=[1])

nl_x = NLS([poly1, poly2, poly3, poly4,poly5])  # nls in state eq
nl_y = NLS()       # nls in output eq

nlsys = NLSS(nl_x,nl_y,sys)
nlsys.output(u=[1,2,3])


R, npp = 1,10
x = np.arange(20).reshape(10,2)
y = np.arange(10).reshape(10,1)
u = np.ones(10).reshape(10,1)
nlsys.x_mod = x
nlsys.y_mod = y
nlsys.um = u
nlsys.R, nlsys.npp = R,npp
nlsys.idx_trans = np.arange(R*npp)
nlsys.idx_remtrans = np.s_[:]
x0 = nlsys.flatten()
jacobian(x0, nlsys)
##direction =
#
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
