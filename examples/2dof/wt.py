#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import namedtuple

from pyvib.morletWT import WT
from pyvib.signal import Signal

path = 'data/'
filename = 'pyds_sweepvrms'
Nm = namedtuple('Nm', 'y yd ydd u t finst fs ns')
sweep_l = pickle.load(open(path + filename + '0.01' + '.pkl', 'rb'))
sweep_h = pickle.load(open(path + filename + '2'  + '.pkl', 'rb'))


f1 = 1e-3
f2 = 10/2/np.pi
nf = 50
f00 = 5
dof = 0
sca = 2*np.pi

lin = Signal(sweep_l.u, sweep_l.fs, sweep_l.ydd)
wtlin = WT(lin)
wtlin.morlet(f1, f2, nf, f00, dof=0)

nlin = Signal(sweep_h.u, sweep_h.fs, sweep_h.ydd)
wtnlin = WT(nlin)
wtnlin.morlet(f1, f2, nf, f00, dof=0)

dof = 0
plt.figure()
plt.plot(sweep_h.finst*sca,sweep_h.y[dof])
plt.title('Sweep')
plt.xlabel('Frequency')
plt.ylabel('Amplitude (m)')


fwlin, ax = wtlin.plot(fss=sweep_l.finst,sca=sca)
# ax.set_xlim([0,20])
# ax.set_ylim([0,20])

fwnlin, ax = wtnlin.plot(fss=sweep_h.finst,sca=sca)
# ax.set_xlim([0,20])
# ax.set_ylim([0,20])

plt.show()
