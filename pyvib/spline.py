#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve, norm

def spline(d, nspline):
    """The inputs are:

    * d: the displacement (or velocity) time series.
    * n: the number of spline segments (or, the number of knots - 1).

    The outputs are:

    * xs: the sought basis functions.
    * kn: the knot locations.
    * dx: this output is basically useless for your use.
    """

    ns = len(d)
    nk = nspline+1

    maxd = 1.01*max(d)
    mind = 1.01*min(d)
    kn = np.linspace(mind, maxd, nspline+1)

    # index of kn before zero
    j0 = np.where(kn <= 0)[0][-1]
    Q = np.zeros((nk,nk))

    for j in range(nk):
        if j == j0:
            t0 = -kn[j]/(kn[j+1]-kn[j])

            Q[0,j] = (t0**3-2*t0**2+t0) * (kn[j+1]-kn[j])
            Q[0,j+1] = (t0**3-t0**2) * (kn[j+1]-kn[j])

            Q[nk-1,j] = (3*t0**2-4*t0+1) * (kn[j+1]-kn[j])
            Q[nk-1,j+1] = (3*t0**2-2*t0) * (kn[j+1]-kn[j])

            if j != 0:
                Q[j,j-1] = 1/(kn[j]-kn[j-1])
                Q[j,j] = 2 * (1/(kn[j]-kn[j-1]) + 1/(kn[j+1]-kn[j]))
                Q[j,j+1] = 1/(kn[j+1]-kn[j])

        elif j != 0 and j != j0 and j != nk-1:
            Q[j,j-1] = 1/(kn[j]-kn[j-1])
            Q[j,j] = 2*(1/(kn[j]-kn[j-1]) + 1/(kn[j+1]-kn[j]))
            Q[j,j+1] = 1/(kn[j+1]-kn[j])

    S = np.zeros((nk,nk))
    for j in range(nk):
        if j == j0:
            t0 = -kn[j]/(kn[j+1]-kn[j])

            S[0,j] = -(2*t0**3-3*t0**2+1)
            S[0,j+1] = -(-2*t0**3+3*t0**2)

            S[nk-1,j] = 6*(t0-t0**2)
            S[nk-1,j+1] = -6*(t0-t0**2)

            if j != 0:
                S[j,j-1] = -3/(kn[j]-kn[j-1])**2
                S[j,j] = 3 * (1/(kn[j]-kn[j-1])**2 - 1/(kn[j+1]-kn[j])**2)
                S[j,j+1] = 3/(kn[j+1]-kn[j])**2

        elif j != 0 and j != j0 and j != nk-1:
            S[j,j-1] = -3/(kn[j]-kn[j-1])**2
            S[j,j] = 3*(1/(kn[j]-kn[j-1])**2 - 1/(kn[j+1]-kn[j])**2)
            S[j,j+1] = 3/(kn[j+1]-kn[j])**2

    dx = solve(Q, S)
    x = np.zeros((nspline,nk,ns))

    for j in range(nspline):

        u = np.zeros(ns)
        t = np.zeros(ns)
        t1 = np.zeros(ns)

        if j == 0:
            mask = (kn[j] <= d) & (d <= kn[j+1])
        else:
            mask = (kn[j] < d) & (d <= kn[j+1])

        u[mask] = d[mask]
        t[mask] = (u[mask]-kn[j]) / (kn[j+1]-kn[j])
        t1[mask] = 1

        A = 2*t**3 - 3*t**2 + t1
        B = -2*t**3 + 3*t**2
        C = (t**3-2*t**2+t) * (kn[j+1]-kn[j])
        D = (t**3-t**2) * (kn[j+1]-kn[j])

        x[j,j,:] += A
        x[j,j+1,:] += B

        for k in range(nk):
            x[j,k,:] += C*dx[j,k] + D*dx[j+1,k]

    xs = np.squeeze(np.sum(x,0))
    return xs, kn, dx


# fs = 20
# x = np.arange(1000)/fs
# y = np.sin(x)
# xs, kn, dx = spline(y,5)
