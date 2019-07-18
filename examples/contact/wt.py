#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import namedtuple

from pyvib.morletWT import WT
from pyvib.signal import Signal


path = 'data/'
filename = 'pyds_sweepvrms0.03'
Nm = namedtuple('Nm', 'y yd ydd u t finst fs')
sweep1 = pickle.load(open(path + filename + '.pkl', 'rb'))
#sweep2 = pickle.load(open(path + 'sweep2.pkl', 'rb'))


nlin = Signal(sweep1.u, sweep1.fs, sweep1.ydd)
nlin.set_signal(y=sweep1.y, yd=sweep1.yd, ydd=sweep1.ydd)

f1 = 1e-3
f2 = 10/2/np.pi
nf = 100
f00 = 7
dof = 0
sca = 2*np.pi

wtnlin = WT(nlin)
wtnlin.morlet(f1, f2, nf, f00, dof=0)
fwnlin, ax = wtnlin.plot(sweep1.finst, sca)

plt.show()
