#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pickle

from collections import namedtuple

from pyvib.morletWT import WT
from pyvib.signal import Signal

Nm = namedtuple('Nm', 'y yd ydd u t finst fs')
sweep1 = pickle.load(open('data/sweep1.pkl', 'rb'))
#sweep2 = pickle.load(open(path + 'sweep2.pkl', 'rb'))

# dof where nonlinearity is
nldof = 27

# wt parameters
nf = 100
f00 = 10

# Load nonlinear signal
snlin = Signal(sweep1.u, sweep1.fs, sweep1.ydd)
wtnlin = WT(snlin)
wtnlin.morlet(f1=5, f2=300, nf=nf, f00=f00, dof=nldof)
fwnlin, ax = wtnlin.plot(fss=sweep1.finst)
#ax.set_xlim([0,20])
#ax.set_ylim([0,20])

plt.show()
