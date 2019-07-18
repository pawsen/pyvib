#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import PyDSTool as dst
from matplotlib import pyplot as plt
import numpy as np
import time
from collections import namedtuple
import pickle
from pyvib.import forcing

savedata = True
saveacc = True

"""
The discontinuous stiffness at impact distance b, is given by
alpha: the slope in-between impacts,
beta: the slope after impact.
g: multiplication of the discontinuous force, ie. with
alpha = 0, beta = 1 and g = 3, then the slope after impacts is 3.
"""

y0, v0 = 0, 0
alpha = 0
beta = 1
b = 0.1745
alpha = 0
beta = 1
g = 3
vsweep = 0.01

ftype = 'sweep'
ftype = 'multisine'

targetlang = 'python'
targetlang = 'c'

# ns = 30000
# fs =40*f2
if ftype == 'multisine':
    vrms = 0.2
    nrep = 10
    ns = 15000
    f1 = 1e-3/2/np.pi
    f2 = 10/2/np.pi
    fs = 20*f2
    u, t_ran = forcing.randomPeriodic(vrms,fs, f1,f2,ns=ns, nrep=nrep)
    saveacc = False
    finst = 0
elif ftype == 'sweep':
    nrep = 1
    vrms = 0.03
    f1 = 1e-3/2/np.pi
    f2 = 2/2/np.pi
    fs = 40*f2
    inctype='log'
    inctype ='lin'
    u, t_ran, finst = forcing.sineSweep(vrms,fs, f1,f2,vsweep, nrep, inctype)
    # we dont use the first value, as it is not repeated when the force is
    # generated. This is taken care of in randomPeriodic.
    ns = (len(u)-1) // nrep
elif ftype == 'sine':
    u, t_ran = forcing.sineForce(vrms, f1, fs, nsper=ns)
else:
    raise ValueError('Wrong type of forcing', ftype)

print(ftype)
print(ns)

abseps = 1e-9
xData = {'force': u}
my_input = dst.InterpolateTable({'tdata': t_ran,
                                 'ics': xData,
                                 'name': 'interp1d',
                                 'method': 'linear',  # next 3 not necessary
                                 'checklevel': 1,
                                 'abseps': 1e-12,
                              }).compute('interp1d')

DSargs = dst.args(name='duffing_sweep')
tdomain = [t_ran[0], t_ran[-1]]
DSargs.tdata = tdomain
DSargs.inputs = my_input.variables['force']
DSargs.pars = {'m':1, 'c':0.045*2, 'k':1.0, 'alpha':alpha, 'beta':beta,
               'b':b, 'g':g}

DSargs.varspecs = {'y': 'v',
                   'v': \
                   '(-k * y' \
                   '-c*v ' \
                   '-f(y) + force) / m',
                   'inval': 'force'}
DSargs.vars = ['y', 'v']

# Piecewise linear function:
DSargs.fnspecs = {
    'f': (['y'], '(y+(beta-alpha)*(abs(y-b) - abs(y+b))/2)*g')
}

DSargs.ics = {'y': y0, 'v': v0}
DSargs.algparams = {'max_step': 0.01,'rtol':1e-14, 'max_pts': 3000000,'refine':
                    0}#, 'jac_recompute':1e-6}
DSargs.checklevel = 2

if targetlang == 'python':
    DS = dst.Generator.Vode_ODEsystem(DSargs)
else: # use radau for stiff problems
    #DS = dst.Generator.Dopri_ODEsystem(DSargs)
    DS = dst.Generator.Radau_ODEsystem(DSargs)

startTime = time.time()

DS.set(#tdata=[t0, t1],
       tdata=[t_ran[0], t_ran[-1]],
       ics={'y':y0, 'v':v0})
traj = DS.compute('in-table')
pts = traj.sample(dt=1/fs, precise=True)

print('Integration done in: {}'.format(time.time()-startTime))

a = 0
y = pts['y']
v = pts['v']
t = pts['t']
u_pol = pts['inval']

def recover_acc(t, y, v):
    """Recover the acceleration from the RHS:
    """
    ns = len(y)
    a = np.empty(y.shape)
    for i in range(ns):
        a[i] = DS.Rhs(t[i], {'y':y[i], 'v':v[i]}, DS.pars)[0]
    print('accelerations recovered')
    return a
if saveacc:
    a = recover_acc(t, y, v)


relpath = 'data/' + 'pyds_' + ftype + 'vrms' + str(vrms)
Nm = namedtuple('Nm', 'y yd ydd u_pol t finst fs ns')
sweep1 = Nm(y,v,a,u,t,finst,fs,ns)
if savedata:
    pickle.dump(sweep1, open(relpath + '.pkl', 'wb'))
    print('data saved as {}'.format(relpath))


plt.figure(1)
plt.clf()
plt.plot(t, y, '-k') #, label = 'disp')
plt.xlabel('Time (t)')
plt.ylabel('Displacement (m)')

plt.show()
# plt.legend()




# Does not really work. I do not know how to update fnspecs on impact
# impact1_args = {'eventtol': abseps/10,
#            'eventdelay': abseps*10,
#            'eventinterval': abseps*10,
#            'active': True,
#            'term': False,
#            'precise': True,
#            'name': 'impact1'}
# impact2_args = {'eventtol': abseps/10,
#            'eventdelay': abseps*10,
#            'eventinterval': abseps*10,
#            'active': True,
#            'term': False,
#            'precise': True,
#            'name': 'impact2'}

# # Upper: y-0.1',1
# # Low y+0.1',-1,
# impact_low_ev = dst.Events.makeZeroCrossEvent('y+0.1',-1,
#                                               impact1_args,
#                                               ['y'],
#                                               ['b', 'beta'],
#                                               #fnspecs={'f': (['y'], '0')},
#                                            #fnspecs={'f': (['y'], '5*beta*y')},
#                                            fnspecs={'f':(['y'], '0')},
#                                            targetlang=targetlang)
# impact_middle_ev = dst.Events.makeZeroCrossEvent('y+0.1',1,
#                                            impact2_args,
#                                            ['y'],
#                                            ['b', 'beta'],
#                                            fnspecs={'f': (['y'], '0')},
#                                            targetlang=targetlang)

#DSargs.events = [impact_low_ev, impact_middle_ev]
