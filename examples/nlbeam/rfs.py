#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import namedtuple

from pyvib.signal import Signal
from pyvib.rfs import RFS

Nm = namedtuple('Nm', 'y yd ydd u t finst fs')
sweep1 = pickle.load(open('data/sweep1.pkl', 'rb'))
#sweep2 = pickle.load(open(path + 'sweep2.pkl', 'rb'))

signal = Signal(sweep1.u, sweep1.fs, sweep1.ydd)
signal.set_signal(y=sweep1.y, yd=sweep1.yd, ydd=sweep1.ydd)

rfs = RFS(signal,dof=27)
rfs.plot()

plt.show()
