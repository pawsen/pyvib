#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

"""
Parameters for 2DOF duffing system

    #              x1                          x2                        #
    #            +-->                        +-->                        #
    #            |                           |                           #
    #  d1 __     +--------------+  d2 __     +--------------+  d3 __     #
    #-----__|----|              |-----__|----|              |-----__|----#
    #  k1        |              |  k3        |              |  k4        #
    #__/\  /\  __|      M1      |__/\  /\  __|       M2     |__/\  /\  __#
    #    \/  \/  |              |    \/  \/  |              |    \/  \/  #
    #  k2 ^      |              |            |              | k2 ^       #
    #__/\/ /\  __|              |            |              |__/\/ /\  __#
    #   /\/  \/  +--------------+            +--------------+   /\/  \/  #

Mode & Frequency (rad/s) & Damping ratio (%)
1    & 1.00              &       5.00
2    & 3.32              &       1.51
"""

m1 = 1    # kg
m2 = 1
k1 = 1    # N/m
k2 = 5
k3 = 1
c1 = 0.1  # N/ms
c2 = 0.1

mu1 = 1    # N/m^3
mu2 = 1    # N/m^3

fdof = 0
vrms = 2  # N
f0 = 1.2/2/np.pi  # Hz
f1 = 1e-4/2/np.pi
f2 = 5/2/np.pi
fs = 10
vsweep = 5
nper = 1
nsper = 1000
inctype = 'log'

M = np.array([[m1,0],[0,m2]])
C = np.array([[c1,0],[0,c2]])
K = np.array([[k1+k2, -k2],[-k2, k2+k3]])
M, C, K = np.atleast_2d(M,C,K)

inl = np.array([[0,-1],
                [1,-1]])
enl = np.array([3,3,2,2])
knl = np.array([mu1, mu2])

par = {
    'M': M,
    'C': C,
    'K': K,
    'fdof': fdof,
    'vrms': vrms,
    'f0': f0,
    'f1': f1,
    'f2': f2,
    'inl': inl,
    'enl': enl,
    'knl': knl,
    'fs': fs,
    'vsweep': vsweep,
    'nper': nper,
    'nsper': nsper,
    'inctype': inctype,
    'm1': m1,
    'm2': m2,
    'c1': c1,
    'c2': c2,
    'k1': k1,
    'k2': k2,
    'k3': k3,
    'mu1': mu1,
    'mu2': mu2
}

