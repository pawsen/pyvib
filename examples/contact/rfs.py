#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import namedtuple

from pyvib.signal import Signal
from pyvib.rfs import RFS

path = 'data/'
filename = 'pyds_sweepvrms0.03'
Nm = namedtuple('Nm', 'y yd ydd u t finst fs')
sweep1 = pickle.load(open(path + filename + '.pkl', 'rb'))
#sweep2 = pickle.load(open(path + 'sweep2.pkl', 'rb'))


signal = Signal(sweep1.u, sweep1.fs, sweep1.ydd)
signal.set_signal(y=sweep1.y, yd=sweep1.yd, ydd=sweep1.ydd)

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
