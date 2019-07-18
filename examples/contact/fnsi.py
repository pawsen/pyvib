#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pickle
from collections import namedtuple

from pyvib.signal import Signal
from pyvib.fnsi import FNSI
from pyvib.common import modal_properties, db, frf_mkc
from pyvib.helper.fnsi_plots import (plot_modes, plot_knl, plot_linfrf,
                                   plot_stab, plot_svg)
from pyvib.frf import FRF
from pyvib.fnsi import NL_force, NL_polynomial, NL_spline
from pyvib.interpolate import piecewise_linear




sca = 2*np.pi


ftype = 'multisine'
filename = 'data/' + 'pyds_' + ftype + 'vrms0.2'
Nm = namedtuple('Nm', 'y yd ydd u t finst fs ns')
msine = pickle.load(open(filename + '.pkl', 'rb'))

# which dof to get H1 estimate from/show periodicity for
dof = 0

fmin = 0
fmax = 10/2/np.pi
nlin = Signal(msine.u, msine.fs, y=msine.y)
fper, ax = nlin.periodicity(msine.ns, dof, offset=0)


per = [7,9]
nlin.cut(msine.ns, per)

isnoise = False

inl = np.array([[0,-1]])
nl_spline = NL_spline(inl, nspl=15)
nl = NL_force()
nl.add(nl_spline)
iu = 0
idof = [0]
nldof = []

ims = 60
nmax = 40
ncur = 6
nlist = np.arange(2, nmax+3, 2)
dof = 0

# nonlinear identification
fnsi = FNSI(nlin, nl, idof, fmin, fmax)
fnsi.calc_EY()
fnsi.svd_comp(ims)
sd = fnsi.stabilisation_diagram(nlist)
fnsi.id(ncur)
fnsi.nl_coeff(iu, dof)
fsdlin, ax = plot_stab(fnsi, nlist, sca)

# Linear identification at high level
nl = NL_force()
fnsi2 = FNSI(nlin, nl, idof, fmin, fmax)
fnsi2.calc_EY()
fnsi2.svd_comp(ims)
fnsi2.id(ncur)
fnsi2.nl_coeff(iu, dof)

# Linear identification at low level
filename = 'data/' + 'pyds_' + ftype + 'vrms0.005'
msine_lin = pickle.load(open(filename + '.pkl', 'rb'))
lin = Signal(msine_lin.u, msine_lin.fs, y=msine_lin.y)
per = [7]
lin.cut(msine_lin.ns, per)
nl = NL_force()
fnsi_lin = FNSI(lin, nl, idof, fmin, fmax)
fnsi_lin.calc_EY()
fnsi_lin.svd_comp(ims)
fnsi_lin.id(ncur)
fnsi_lin.nl_coeff(iu, dof)


# identified spline knots
if len(fnsi.nonlin.nls) > 0:
    knl_c = np.mean(fnsi.nonlin.nls[0].knl,axis=1)
    knl = np.real(knl_c)
    kn = fnsi.nonlin.nls[0].kn
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(kn, knl)
    x = np.linspace(min(kn),max(kn), 1000)

    plt.figure(2)
    plt.clf()
    plt.plot(x,cs(x),'-k')
    plt.plot(kn, knl, 'xk')
    b = 0.1745
    plt.axvline(-b,c='k', ls='--')
    plt.axvline(b,c='k', ls='--')

    b = 0.1745
    alpha = 0
    beta = 3
    slope = np.array([beta, alpha, beta])
    xk = np.array([-b, b])
    yk = np.array([0, 0]) #np.array([-b, b])
    y2 = piecewise_linear(xk,yk,slope,delta=None,xv=x)
    plt.plot(x,y2,'--')

    plt.xlabel('Displacement')
    plt.ylabel('Nonlinear restoring force')
    figrf = plt.gcf()

def print_modal(fnsi):
    # calculate modal parameters
    modal = modal_properties(fnsi.A, fnsi.C)
    natfreq = modal['wn']
    dampfreq = modal['wd']
    damping = modal['zeta']
    nw = min(len(natfreq), 8)

    print('Undamped ω: {}'.format(natfreq[:nw]*sca))
    print('damped ω: {}'.format(dampfreq[:nw]*sca))
    print('damping: {}'.format(damping[:nw]))
    return modal

# FRF
frf = FRF(nlin, fmin=1e-3, fmax=5/2/np.pi)
frf_freq, frf_H = frf.periodic()

print('## linear identified at low level')
print_modal(fnsi_lin)
print('## linear identified at high level')
print_modal(fnsi2)
print('## nonlinear identified at high level')
print_modal(fnsi)

fH1, ax = plt.subplots()
ax.plot(frf_freq*sca, db(np.abs(frf_H[dof])), '-.k', label='From signal')
plot_linfrf(fnsi_lin, dof, sca=sca, fig=fH1, ax=ax, label='lin low level')
plot_linfrf(fnsi, dof, sca=sca, fig=fH1, ax=ax, label='nl high level')
plot_linfrf(fnsi2, dof, sca=sca, fig=fH1, ax=ax, ls='--', label='lin high level')
# ax.set_title(''); ax.legend_ = None
ax.legend()

plt.show()
