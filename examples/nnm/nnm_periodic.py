#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
from scipy.integrate import odeint


T = 2.0215
ns = 100
x0= 4.9009
v0 = 0
M = 1
k = 1
k3 = 0.5

points = [[4.9009, 0], [-1.0313, -12.9188], [-2.9259, 11.8894]]
p = [1, 5, 3]
lw = 1

def system_1dof(w, t):
    """SDOF Duffing oscillator for odeint

    w : state vector
    t : current time step
    f : Return system matrix
    """
    x1, y1 = w
    # Create f = (x1',y1')
    f = [y1,
         (-k*x1 - k3*x1**3)/M]
    return f


x0_vec = np.array([1, 0.9, 1.1]) * x0
sol_vec = []
t = np.arange(ns+1)/ns*T

for x0_ in x0_vec:
    w0 = x0_, v0
    sol = odeint(system_1dof, w0, t)
    sol_vec.append(sol)

print(sol_vec[0][ns//p[0],0], sol_vec[0][ns//p[0],1],
      sol_vec[0][ns//p[1]*4,0], sol_vec[0][ns//p[1]*4,1],
      sol_vec[0][ns//p[2],0], sol_vec[0][ns//p[2],1]
      )


plt.ion()
plt.figure(1)
plt.clf()
plt.xlabel('Time (s)')
plt.ylabel(r'$x$')
plt.grid(True)
plt.plot(t, sol_vec[0][:,0], 'k', linewidth=lw, label=r'$x_1$')
plt.plot(t, sol_vec[1][:,0], 'k--', linewidth=lw, label=r'$x_2$')
plt.plot(t, sol_vec[2][:,0], 'k-.', linewidth=lw, label=r'$x_2$')
plt.plot(t[0],sol_vec[0][0,0],'dk', mfc='none')
plt.plot(t[ns//p[0]],sol_vec[0][ns//p[0],0],'dk', mfc='none')
plt.plot(t[ns//p[1]*4],sol_vec[0][ns//p[1]*4,0],'sk', mfc='none')
plt.plot(t[ns//p[2]],sol_vec[0][ns//p[2],0],'ok', mfc='none')
# plt.legend()
plt.tight_layout()
fig1 = plt.gcf()

plt.figure(2)
plt.clf()
plt.plot(sol_vec[0][:,0], sol_vec[0][:,1], 'k', linewidth=lw, label=r'$x_1$')
plt.plot(sol_vec[1][:,0], sol_vec[1][:,1], 'k--', linewidth=lw, label=r'$x_2$')
plt.plot(sol_vec[2][:,0], sol_vec[2][:,1], 'k-.', linewidth=lw, label=r'$x_2$')
plt.plot(sol_vec[0][ns//p[0],0], sol_vec[0][ns//p[0],1],'dk', mfc='none')
plt.plot(sol_vec[0][ns//p[1]*4,0], sol_vec[0][ns//p[1]*4,1],'sk', mfc='none')
plt.plot(sol_vec[0][ns//p[2],0], sol_vec[0][ns//p[2],1],'ok', mfc='none')
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot x$')
# plt.axis('equal')
plt.tight_layout()
fig2 = plt.gcf()


plt.show()
