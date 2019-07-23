#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

from pyvib.rfs import RFS
from pyvib.signal import Signal2 as Signal

path = 'data/'
filename = 'pyds_sweepvrms0.03'
Nm = namedtuple('Nm', 'y yd ydd u t finst fs')
sweep1 = pickle.load(open(path + filename + '.pkl', 'rb'))
#sweep2 = pickle.load(open(path + 'sweep2.pkl', 'rb'))

y=sweep1.y
ydd = sweep1.ydd
yd = sweep1.yd

try:
    raise TypeError('Uncomment this line to do numerical differentiation')
    fs = sweep1.fs
    # select only part of the signal for better differentiation using
    # regularization
    low_idx, high_idx = int(1000*fs), int(1600*fs)
    t = np.arange(len(y))/fs
    y = y[low_idx:high_idx]
    ydd = sweep1.ydd[low_idx:high_idx]
    t = t[low_idx:high_idx]

    # this library breaks because of too high memory usage
#    import sys
#    # https://pythonhosted.org/scikits.datasmooth/regularsmooth.html
#    sys.path.append('/home/paw/src/scikit-datasmooth')
#    from scikits import datasmooth as ds
#    yd = ds.calc_derivative(t,y,d=1)

    # noise-free data. Just do it simple
    yd = np.gradient(y, 1/fs)
    ydd = np.gradient(yd, 1/fs)

    # differentiate using total variation regularization
    # https://github.com/pawsen/tvregdiff
    # import sys
    # sys.path.append('/home/paw/src/tvregdiff')
    # from tvregdiff import TVRegDiff
    # yd = TVRegDiff(y, itern=200, alph=0.1, dx=1/fs, ep=1e-2, 
    #                scale='large', plotflag=0)
    # ydd = TVRegDiff(yd, itern=200, alph=1e-1, dx=1/fs, ep=1e-2,
    #                scale='large', plotflag=0)
except Exception as err:
    import traceback
    traceback.print_exc()
    print(err)

signal = Signal(sweep1.u, sweep1.fs, sweep1.ydd)
signal.set_signal(y=y, yd=yd, ydd=ydd)

rfs = RFS(signal,dof=0)
rfs.plot()

## Extract subplot
frfs = rfs.plot.fig
b = 0.1745
ax2 = rfs.plot.ax2d
ax2.axvline(-b,c='k', ls='--')
ax2.axvline(b,c='k', ls='--')
fig2 = plt.figure()
ax2.figure=fig2
fig2.axes.append(ax2)
fig2.add_axes(ax2)

dummy = fig2.add_subplot(111)
ax2.set_position(dummy.get_position())
dummy.remove()
ax2.set_title('')

plt.show()
