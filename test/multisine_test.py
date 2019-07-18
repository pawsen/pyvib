#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from vib.forcing import multisine
from vib.common import db

"""Simple example of quantifying odd/even nonlinearities using multisine.

The nonparametric analysis/quantifying only works for one period, P=1.
The nonlinearities become visible at the unexcited frequencies by using
ftype={odd, oddrandom}.
"""

fs = 4096  # sampling frequency, Hz
N = 4096  # number of points per period
P = 1  # number of periods
M = 1  # number of realizations
f1 = 1  # lowest excited frequency
f2 = 100  # highest excited frequency
uStd = 1  # standard deviation of the generated signal

u, t, lines, freq = multisine(f1, f2, fs, N, P, M, ftype='oddrandom', std=uStd)

# apply input to a static nonlinear system
y = u + 0.001*u**2 - 0.1*u**3
# perform nonlinear analysis
Y = fft(y)
# separate even and odd lines. 1 is dc, ie. 1+2n is even
lines_even = np.arange(0,N,2)
lines_odd_det = np.arange(1,N,2)
lines_odd_det = np.setdiff1d(lines_odd_det, lines)

plt.ion()
# signal plots
plt.figure(1)
plt.clf()
plt.plot(t, u[0])
plt.xlabel('time (s)')
plt.ylabel('magnitude')
plt.title('Multisine: {} realizations of {} periods of {} '
          'samples per period'.format(M,P,N))

plt.figure(2)
plt.clf()
plt.subplot(2,1,1)
plt.plot(freq,db(fft(u[0,:N])),'-*')
plt.xlabel('frequency (Hz)')
plt.ylabel('magnitude (dB)')
plt.title('FFT of one period of the multisine realizations')

plt.subplot(2,1,2)
plt.plot(freq,np.angle(fft(u[0,:N])),'-*')
plt.xlabel('frequency (Hz)')
plt.ylabel('phase (rad)')
plt.title('FFT of one period of the multisine realizations')
plt.tight_layout()

# nl-test plot
plt.figure(3)
plt.clf()
plt.plot(u[0],y[0],'.')
plt.xlabel('input')
plt.ylabel('output')
plt.title('Static Nonlinear Function')

plt.figure(4)
plt.clf()
plt.plot(freq[lines],db(Y[0,lines]),'*')
plt.plot(freq[lines_odd_det],db(Y[0,lines_odd_det]),'^')
plt.plot(freq[lines_even],db(Y[0,lines_even]),'o')
plt.xlim([0,fs/2])
plt.xlabel('frequency (Hz)')
plt.ylabel('magnitude (dB)')
plt.title('FFT output')
plt.legend(('excited lines','odd detection lines','even detection lines'))


plt.show()
