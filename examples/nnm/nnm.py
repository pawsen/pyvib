#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class System(object):
    """
    Abstract class for systems.
    Defines plot and integration method.
    """
    def __init__(self, p, w0, figname=None):
        self.p = p
        self.w0 = w0
        self.w_desc = 'x1 ,x2, y1, y2'
        print("Creating {} with parameters ({})={} and init ({})={}".
              format(self.systype, self.p_desc, self.p, self.w_desc, self.w0))
        if figname is not None:
            self.figname = figname


    def systype(self):
        pass

    def sysdef(self):
        pass

    def integrate(self, w0=None):
        """Notice:
        scipy odeint and ode has the order of inputs (w,t) reversed.
        """
        if w0 is not None:
            # update init cond
            self.w0 = w0
            print("updated {} with init {}".format(self.systype, self.w0))

        # ODE solver parameters
        abserr = 1.0e-8
        relerr = 1.0e-6
        stoptime = 10
        numpoints = 1000
        # w0 = init_cond(state)
        self.t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]
        # Call the ODE solver.
        self.wsol = odeint(self.sysdef, self.w0, self.t,
                           atol=abserr, rtol=relerr)

    def plot(self, save = False, figname = None):
        # unpack solution
        x1, x2, y1, y2 = self.wsol.T
        lw = 1

        plt.ion()
        plt.figure(1), plt.clf()
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement (m)')
        plt.grid(True)
        plt.hold(True)
        plt.plot(self.t, x1, 'b', linewidth=lw, label=r'$x_1$')
        plt.plot(self.t, x2, 'g--', linewidth=lw, label=r'$x_2$')
        plt.legend()
        plt.tight_layout()
        if save is True:
            plt.savefig('plots/nnm/{}_time_hist.pdf'.format(self.figname))

        plt.figure(2), plt.clf()
        plt.plot(x1, x2, 'b', linewidth=lw)
        plt.xlabel(r'Displacement $x_1$ (m)')
        plt.ylabel(r'Displacement $x_2$ (m)')
        # plt.axis('equal')
        plt.tight_layout()
        if save is True:
            plt.savefig('plots/nnm/{}_conf_space.pdf'.format(self.figname))
        plt.show()


class nonlin(System):
    """
    Two mass system with cubic stiffness between left mass and wall

    Arguments:
        w :  vector of the state variables:
            w = [x1,x2,y1,y2]
        t :  time
        p :  vector of the parameters:
            p = [m1,m2,k1,k2,k3,d1,d2,d3]

    #              x1                          x2                        #
    #            +-->                        +-->                        #
    #            |                           |                           #
    #  d1 __     +--------------+  d2 __     +--------------+  d3 __     #
    #-----__|----|              |-----__|----|              |-----__|----#
    #  k1        |              |  k3        |              |  k4        #
    #__/\  /\  __|      M1      |__/\  /\  __|       M2     |__/\  /\  __#
    #    \/  \/  |              |    \/  \/  |              |    \/  \/  #
    #  k2 ^      |              |            |              |            #
    #__/\/ /\  __|              |            |              |            #
    #   /\/  \/  +--------------+            +--------------+            #

    """
    def __init__(self, p, w0, figname = None):
        self.systype = 'nonlin'
        self.p_desc = 'm1, m2, k1, k2, k3, k4'
        System.__init__(self, p, w0, figname)

    def sysdef(self, w, t):
        x1, x2, y1, y2 = w
        m1, m2, k1, k2, k3, k4 = self.p

        # Create f = (x1',x2',y1',y2')
        f = [y1,
             y2,
             (-(k1 + k3) * x1 + k3 * x2 - k2 * x1**3) / m1,
             (k3 * x1 - (k3 + k4) * x2) / m2]
        return f


class linear(System):
    """
    Underlying linear system
    Init-cond are given by:

    x1 = c*x2
    """
    def __init__(self, p, w0, figname = None):
        self.systype = 'linear'
        self.p_desc = 'm1, m2, k1, k3, k4'
        System.__init__(self, p, w0, figname)

    def sysdef(self, w, t):
        x1, x2, y1, y2 = w
        m1, m2, k1, k3, k4 = self.p

        # Create f = (x1',y1',x2',y2')
        f = [y1,
             y2,
             (-x1 - k3*(x1 - x2) - k1*x1**3) / m1,
             (-x2 + k3*(x1 - x2) - k4*x2**3) / m2]
        return f


class similar(System):
    """
    Symmetric system, resulting in only similar NNMs.
    Init-cond are given by similar_nnm(K)
    K is the coupling strength.
    The localization of motion increases as K -> 0.
    """
    def __init__(self, p, w0, figname = None):
        self.systype = 'similar'
        self.p_desc = 'm1, m2, K'
        System.__init__(self, p, w0, figname)

    def sysdef(self, w, t):
        x1, x2, y1, y2 = w
        m1, m2, K = p

        # Create f = (x1',y1',x2',y2')
        f = [y1,
             y2,
             (-x1 - K*(x1 - x2)**3 - x1**3) / m1,
             (-x2 + K*(x1 - x2)**3 - x2**3) / m2]
        return f

    def nnm(self, K = None):
        """
        Init cond for the similar system.
        Returns the c's of K*(1+c)*(c-1)^3 = c*(1-c^2). Note that x1 = c*x2, so for
        easiness we choose x2 = c1 = 1.

        c is found from
        K*(1+c)*(c-1)^3 = c*(1-c^2), where K is the coupling strength.

        For K>1/4, there are only the two real solutions for c=1,-1
        For K<=1/4, there are four real solutions. The NMM for c=-1 becomes unstable.
        """
        if K is None:
            K = self.p[2]
        c1 = 1
        c2 = -1
        c3 = 1/2 * (2*K - 1 + np.sqrt(-4*K+1))/K
        c4 = 1/2 * (2*K - 1 - np.sqrt(-4*K+1))/K

        # update K and init cond. Can also be done by calling integrate with par
        self.p[2] = K
        self.w0 = (1, c3, 0, 0)
        print("updated {} with K {} and init {}".format(self.systype, K, self.w0))


        return c1, c2, c3, c4


# Parameter values
# Masses:
m1, m2 = 1, 1

w = (1, 1, 0, 0)
k1, k3, k4 = 1, 1, 1
p = [m1, m2, k1, k3, k4]
lin_inphase = linear(p, w, 'lin_inphase')

w = (1,  -0.38196601125010526, 0, 0)
lin_inphase2 = linear(p, w, 'lin_inphase2')

w = (1, -1, 0, 0)
k1, k3, k4 = 1, 1, 1
p = [m1, m2, k1, k3, k4]
lin_outphase = linear(p, w, 'lin_outphase')

# internally resonant 3:1
w = (8.476, 54.263, 0, 0)
k1, k2, k3, k4 = 1, 0.5, 1, 1
p = [m1, m2, k1, k2, k3, k4]
nonlin_31 = nonlin(p, w, 'nonlin_31')

w = (3.319, 11.134, 0, 0)
nonlin_inphase = nonlin(p, w, 'nonlin_inphase')

w = (-10.188, 0.262, 0, 0)
nonlin_outphase = nonlin(p, w, 'nonlin_outphase')

w = (1, 1, 0, 0)
k = 0.2
p = [m1, m2, k]
similar1 = similar(p, w, 'similar1')
# update init to bifurcated NNM
similar1.nnm()

# 
w = (1, -1, 0, 0)
similar2 = similar(p, w, 'similar2')

# for sys in similar1, lin_inphase2, similar2:
#     sys.integrate()
#     sys.plot()


for sys in lin_inphase, lin_outphase, nonlin_31, nonlin_inphase, nonlin_outphase, similar1, similar2:
    sys.integrate()
    sys.plot(False)
