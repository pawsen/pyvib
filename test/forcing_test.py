#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from matplotlib import pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import forcing
import numpy.testing as npt
from common import db


""" TODO: When nrep != 1, the frequency content is not the same as when nrep =
1. Why? """
vrms = 1
fs = 50
f1 = 5
f2 = 20
ns = 100
nrep = 1

u, t = forcing.randomPeriodic(vrms,fs, f1,f2,ns, nrep)

npt.assert_allclose(np.linalg.norm(t), 11.6335721083)
npt.assert_allclose(np.linalg.norm(u), 10.0498756211)


U = np.fft.fft(u)
idx = len(U)//2

U_plot = db(np.abs(U[0:idx]))
freq = np.fft.fftfreq(len(U), d=1/fs)
freq = freq[0:idx]

fig1 = plt.figure()
plt.clf()
plt.plot(freq, U_plot, '*k' )
plt.axvline(f1,color='k', linestyle='--')
plt.axvline(f2,color='k', linestyle='--')
plt.title('Frequency content for randomPeriodic signal')
plt.xlabel( 'Frequency (Hz)')
plt.ylabel('FFT basis periodic random (dB)')
plt.grid()


amp = 1
fs = 100
f1 = 1
f2 = 5
vsweep = 50
inctypes = ['lin','log']
t0 = 0

norm_t = [60.8105911828, 26.8334474118]
norm_u = [15.5130744548, 11.8222391343]
for i, inctype in enumerate(inctypes):
    u, t, finst = forcing.sineSweep(amp, fs, f1, f2, vsweep, nrep=1, inctype=inctype, t0=t0)

    npt.assert_allclose(np.linalg.norm(t), norm_t[i])
    npt.assert_allclose(np.linalg.norm(u), norm_u[i])


fig2 = plt.figure()
plt.clf()
plt.plot(t, u, '-k' )
plt.ylim([-1.5*amp, 1.5*amp])
plt.title('Time history for sineSweep signal, type: {}'.format(inctype))
plt.xlabel('Time (s)')
plt.ylabel('Sine sweep excitation (N)')
plt.grid()

fig3 = plt.figure()
plt.clf()
plt.plot(finst, u,'-k' )
plt.ylim([ -1.5*amp, 1.5*amp])
plt.title('Frequency history for sineSweep signal, type: {}'.format(inctype))
plt.xlabel('Instantaneous frequency (Hz)')
plt.ylabel( 'Sine sweep excitation (N)')
plt.grid()

fig4 = plt.figure()
plt.clf()
plt.plot(t, finst, '-k' )
plt.title('Frequency history for sineSweep signal, type: {}'.format(inctype))
plt.xlabel('Time (s)')
plt.ylabel('Instantaneous frequency (Hz)')
plt.grid()

U = np.fft.fft(u)
idx = len(U)//2
U_plot = db(np.abs(U[0:idx]))
freq = np.fft.fftfreq(len(U), d=1/fs)
freq = freq[0:idx]

fig5 = plt.figure()
plt.clf()
plt.plot(freq, U_plot, '*k' )
plt.axvline(f1,color='k', linestyle='--')
plt.axvline(f2,color='k', linestyle='--')
plt.title('Frequency content for sineSweep signal, type: {}'.format(inctype))
plt.xlabel( 'Frequency (Hz)')
plt.ylabel('FFT basis periodic random (dB)')
plt.grid()

fig1.savefig('freq_multisine.png')
fig5.savefig('freq_sinesweep.png')

plt.show()
