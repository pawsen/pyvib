#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import namedtuple

from pyvib.signal import Signal
from pyvib.rfs import RFS

path = 'data/'
filename = 'pyds_sweepvrms'
Nm = namedtuple('Nm', 'y yd ydd u t finst fs ns')
#sweep_l = pickle.load(open(path + filename + '0.01' + '.pkl', 'rb'))
sweep_h = pickle.load(open(path + filename + '2'  + '.pkl', 'rb'))


signal = Signal(sweep_h.u, sweep_h.fs, sweep_h.ydd)
signal.set_signal(y=sweep_h.y, yd=sweep_h.yd, ydd=sweep_h.ydd)

rfs = RFS(signal, dof=0)
rfs.plot()


# extract force slice/subplot for saving
# ax2 = rfs.plot.ax2d
# fig2 = plt.figure()
# ax2.figure=fig2
# fig2.axes.append(ax2)
# fig2.add_axes(ax2)

# dummy = fig2.add_subplot(111)
# ax2.set_position(dummy.get_position())
# dummy.remove()
# ax2.set_title('')
