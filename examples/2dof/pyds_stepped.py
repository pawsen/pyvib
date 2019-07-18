#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import PyDSTool as dst
import numpy as np
import time
from parameters import par
np.set_printoptions(precision=5)

savedata = True
vrms = 2
y10, y20, v10, v20 = 0, 0, 0, 0

# Set simulation parameters
ntransient = 500    # transient periods of ext excitation
nsteady = 100       # state state periods of ext excitation
nSimTimeSteps = 25  # sampling rate
OMEGA_start = 0.1   # start of excitation freq range
OMEGA_stop = 5    # end of excitation freq range
n_OMEGA = 100       # numbers in excitation freq range

DSargs = dst.args(name='stepped')
DSargs.pars = {key: par[key] for key in ['m1', 'm2', 'c1', 'c2', 'k1', 'k2',
                                         'k3', 'mu1', 'mu2']
}
DSargs.pars.update({
               'q': vrms,
               'OMEGA': 0})

DSargs.varspecs = {'y1': 'v1',
'y2': 'v2',
                   'v1': \
                   '(-(k1 + k2) * y1' \
                   '-c1*v1 + k2 * y2' \
                   '- mu1 * y1**3 - q*cos(OMEGA*t)) / m1',
                   'v2': \
                   '(k2 * y1 - (k2 + k3) * y2 \
                    -c2*v2 - mu2 * y2**3) / m2'
}

DSargs.vars = ['y1', 'v1', 'y2', 'v2']
DSargs.ics = {'y1': y10, 'v1': v10, 'y2': y20, 'v2': v20}
DSargs.algparams = {'max_pts': 1000000}
DSargs.checklevel = 2

python = True
python = False
if python:
    ode = dst.Generator.Vode_ODEsystem(DSargs)
else:
    ode = dst.Generator.Dopri_ODEsystem(DSargs)

# increase/decrease ext excitation freq
if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == 'True':
        increasing = True
    else:
        increasing = False
else:
    increasing = False
    increasing = True

# Sweep OMEGA
if increasing:
    filename = 'data/stepped_inc'
    OMEGA_vec = np.linspace(OMEGA_start, OMEGA_stop, n_OMEGA)
else:
    filename = 'data/stepped_dec'
    OMEGA_vec = np.linspace(OMEGA_stop, OMEGA_start, n_OMEGA)

t_transient = []
t_steady = []
y_transient = []
y_steady = []
v_transient = []
v_steady = []

print('looping OMEGA from %f to %f in %d steps' \
    % (OMEGA_vec[0], OMEGA_vec[-1], n_OMEGA))

startTime = time.time()
i = 0
for OMEGA in OMEGA_vec:

    print('OMEGA=%f' % OMEGA)

    # adjust time domain and timestep:
    t0 = 0.0  # start time for for the simulation
    ttransient = ntransient*2.0*np.pi/OMEGA  # periods of the excitation force
    tstop = ttransient + nsteady*2.0*np.pi/OMEGA  # periods of the excitation force
    dt = 2*np.pi/OMEGA / nSimTimeSteps  # timesteps per period of the excitation force 

    # set excitation frequency and update time doamain
    ode.set(pars={'OMEGA': OMEGA})

    # solve for transient motion:
    ode.set(tdata=[t0, ttransient],
        ics={'y1':y10, 'v1':v10,'y2':y20, 'v2':v20})

    traj_transient = ode.compute('transient')  # integrate ODE
    pts = traj_transient.sample(dt=dt, precise=True)  # sampling data for plotting
    pts_transient = pts
    y10 = pts['y1'][-1]
    v10 = pts['v1'][-1]
    y20 = pts['y2'][-1]
    v20 = pts['v2'][-1]
    t0 = pts['t'][-1]

    # solve for steady state motion:
    ode.set(tdata=[t0, tstop+0.5*dt],
        ics={'y1':y10, 'v1':v10,'y2':y20, 'v2':v20})
    traj_steady = ode.compute('steady')
    pts = traj_steady.sample(dt=dt, precise=True)
    pts_steady = pts

    # update initial conditions
    y10 = pts['y1'][-1]
    v10 = pts['v1'][-1]
    y20 = pts['y2'][-1]
    v20 = pts['v2'][-1]

    # Save time data
    t_transient.append(pts_transient['t'])
    t_steady.append(pts_steady)
    y_transient.append([pts_transient['y1'],pts_transient['y2']])
    y_steady.append([pts_steady['y1'],pts_steady['y2']])
    v_transient.append([pts_transient['v1'],pts_transient['v2']])
    v_steady.append([pts_steady['v1'],pts_steady['v2']])
    ymax = np.max(y_steady[-1],axis=1)
    ymin = np.min(y_steady[-1],axis=1)
    print("max A {} ".format(np.abs(0.5*(ymax-ymin))))

    i += 1

totalTime = time.time()-startTime
print('')
print(' %d Sweeps with %d transient and %d steady periods.' \
    %(len(OMEGA_vec), ntransient, nsteady))
print(' Each period is sampled in %d steps, a total of %d steps' \
    %(nSimTimeSteps, len(OMEGA_vec)*ntransient*nsteady*nSimTimeSteps))
print(' Total time: %f, time per sweep: %f' % (totalTime,totalTime/len(OMEGA_vec)))

if savedata:
    np.savez(
        filename,
        timestep=dt,
        OMEGA_vec=OMEGA_vec,
        #t_transient=t_transient,
        #y_transient=y_transient,
        #v_transient=v_transient,
        t_steady=t_steady,
        y_steady=y_steady,
        v_steady=v_steady
    )

