#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import PyDSTool as dst
from matplotlib import pyplot as plt
import numpy as np
from collections import namedtuple
import pickle
import time

from pyvib.forcing import sineForce, randomPeriodic, sineSweep
import parameters
par = parameters.par

savedata = True

# The vrms (in N) values are chosen as
# vrms = 0.01, linear
# vrms = 2, nonlinear

vrms = 0.01
y10, y20, v10, v20 = 0, 0, 0, 0

ftype = 'multisine'
ftype = 'sweep'

# get external forcing
finst = []
if ftype == 'multisine':
    ns = 25000
    nrep = 9
    f1 = 0/2/np.pi
    f2 = 20/2/np.pi
    fs = 20*f2
    u, t_ran = randomPeriodic(vrms,fs, f1,f2,ns=ns, nrep=nrep)
    saveacc = False
elif ftype == 'sweep':
    saveacc = True
    nrep = 1
    f1 = 1e-3/2/np.pi
    f2 = 5/2/np.pi
    fs = 20*f2
    vsweep = 0.005
    inctype ='lin'
    # change sweep-direction by: sineSweep(vrms,fs, f2,f1,-vsweep, nper, inctype)
    u, t_ran, finst = sineSweep(vrms,fs, f1,f2,vsweep, nrep, inctype)
    # we dont use the first value, as it is not repeated when the force is
    # generated. This is taken care of in randomPeriodic.
    ns = (len(u)-1) // nrep
elif ftype == 'sine':
    ns = 10000
    u, t_ran = sineForce(vrms, f1, fs, nsper=ns)
else:
    raise ValueError('Wrong type of forcing', ftype)

print('\n parameters:')
print('ftype      \t = %s' % ftype)
print('vrms       \t = %f' % vrms)
print('fs         \t = %d' % fs)
print('f1         \t = %f' % f1)
print('f2         \t = %f' % f2)
print('nsper      \t = %d' % ns)
print('nrep       \t = %d' % nrep)
print('ns_tot     \t = %d' % len(u))


# Interpolation-table for the force
xData = {'force': u}
my_input = dst.InterpolateTable({'tdata': t_ran,
                                 'ics': xData,
                                 'name': 'interp1d',
                                 'method': 'linear',  # next 3 not necessary
                                 'checklevel': 1,
                                 'abseps': 1e-6,
                              }).compute('interp1d')

DSargs = dst.args(name='duffing')
tdomain = [t_ran[0], t_ran[-1]]
DSargs.tdata = tdomain
DSargs.inputs = my_input.variables['force']
DSargs.pars = {key: par[key] for key in ['m1', 'm2', 'c1', 'c2', 'k1', 'k2',
                                         'k3', 'mu1', 'mu2']}

DSargs.varspecs = {'y1': 'v1',
                   'y2': 'v2',
                   'v1': \
                   '(-(k1 + k2) * y1' \
                   '-c1*v1 + k2 * y2' \
                   ' - mu1 * y1**3 + force) / m1',
                   'v2': \
                   '(k2 * y1 - (k2 + k3) * y2' \
                   ' -c2*v2 - mu2 * y2**3) / m2',
                   'inval': 'force'}
DSargs.vars = ['y1', 'v1', 'y2', 'v2']
DSargs.ics = {'y1': y10, 'v1': v10, 'y2': y20, 'v2': v20}
DSargs.algparams = {'init_step': 0.01, 'max_step': 0.01, 'max_pts': 2000000}
DSargs.checklevel = 2

python = False
if python:
    DS = dst.Generator.Vode_ODEsystem(DSargs)
else:
    DS = dst.Generator.Dopri_ODEsystem(DSargs)

startTime = time.time()

# in order not to get too many points for a simulation, the simulation is
# splitted up. If simulation points exceed max_pts, pydstool fails with a
# message: 'No trajectory created'
int_time = (t_ran[-1]-t_ran[0])/nrep
t0 = 0
t1 = int_time
y, v, t, u_pol, a = [], [], [], [], []

y = np.empty((2,0))
v = np.empty((2,0))
#for i in range(nrep):
for i in range(1):
    t0, t1 = t_ran[0], t_ran[-1]
    DS.set(tdata=[t0, t1],
           ics={'y1':y10, 'v1':v10,'y2':y20, 'v2':v20})
    traj = DS.compute('in-table')
    pts = traj.sample(dt=1/fs, precise=True)
    # Dont save the last point, as it will be the first point for next round
    y =  np.hstack((y,[pts['y1'][:-1], pts['y2'][:-1]]))
    v =  np.hstack((v,[pts['v1'][:-1], pts['v2'][:-1]]))
    t.extend(pts['t'][:-1])
    u_pol.extend(pts['inval'][:-1])
    y10 = pts['y1'][-1]
    v10 = pts['v1'][-1]
    y20 = pts['y2'][-1]
    v20 = pts['v2'][-1]
    t0 = pts['t'][-1]
    t1 = t0 + int_time

y = np.hstack((y,np.vstack([pts['y1'][-1], pts['y2'][-1]])))
v = np.hstack((v,np.vstack([pts['v1'][-1], pts['v2'][-1]])))
t.extend([pts['t'][-1]])
u_pol.extend([pts['inval'][-1]])

print('Integration done in: {}'.format(time.time()-startTime))

def recover_acc(t, y, v):
    # Recover the acceleration from the RHS:
    n, ns = y.shape
    a = np.empty(y.shape)
    for i in range(ns):
        a[:,i] = DS.Rhs(t[i], {'y1':y[0,i], 'v1':v[0,i], 'y2':y[1,i], 'v2':v[1,i]}, DS.pars)[:n]
    print('accelerations recovered')
    return a

if saveacc:
    a = recover_acc(t, y, v)

Nm = namedtuple('Nm', 'y yd ydd u t finst fs ns')
if savedata:
    data = Nm(y,v,a,u_pol,t,finst,fs,ns)
    filename = 'data/' + 'pyds_' + ftype + 'vrms' + str(vrms)
    pickle.dump(data, open(filename + '.pkl', 'wb'))
    print('data saved as {}'.format(filename))


plt.figure()
plt.plot(t, y[0], '-k', label = r'$x_1$')
plt.plot(t, y[1], '--r', label = r'$x_2$')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (m)')
plt.title('Force type: {}, periods:{:d}'.format(ftype, nrep))
plt.legend()
plt.show()
