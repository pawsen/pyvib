#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.linalg import norm
from pyvib.common import db
from pyvib.frf import covariance

savefig = True

def load(var, amp, fnsi=True):
    # load recorded data
    fnsi = 'FNSI_' if fnsi else ''
    path = 'data/'
    fname = f"{path}SNJP_{var}m_full_{fnsi}{amp}.mat"
    data = sio.loadmat(fname)
    if var == 'u':
        um, fs, flines, P = [data[k] for k in ['um', 'fs', 'flines', 'P']]
        return um, fs.item(), flines.squeeze(), P.item()
    else:
        return data['ym']

# estimation data. Needed for noise calculation
u, fs, lines, P = load('u',100)
y = load('y',100)
NT, R = u.shape
NT, R = y.shape
npp = NT//P
Ptr = 5
u = u.reshape(npp,P,R,order='F').swapaxes(1,2)[:,None,:,Ptr:]
y = y.reshape(npp,P,R,order='F').swapaxes(1,2)[:,None,:,Ptr:]
uest = u[:,:,0,:]
yest = y[:,:,0,:]
Pest = yest.shape[-1]
# noise estimate over Pest periods
covY = covariance(yest[:,:,None])
noise = np.abs(np.sqrt(Pest*covY.squeeze()))

# validation data. Needed for val-plot, where frf of val-data is shown.
npp = 8192
uval_raw, fs, lines, Pval = load('u', 100, fnsi=False)
yval_raw = load('y', 100, fnsi=False)
uval_raw = uval_raw.reshape(npp,Pval,50,order='F').swapaxes(1,2)[:,None]
yval_raw = yval_raw.reshape(npp,Pval,50,order='F').swapaxes(1,2)[:,None]
Rval = uval_raw.shape[2]

def load_sim(fname):
    # load simulated data
    fname = f'sn_jp_{fname}.pkl'
    try:
        return pickle.load(open(fname,'rb'))
    except FileNotFoundError as e:
        print(e)

rms = lambda y: np.sqrt(np.mean(y**2, axis=0))
rms2 = lambda y: np.sqrt(np.mean(y**2, axis=2))
def calc_model(model):
    # calculate statistic from loaded models
    db_val = db(rms2(model['val_err']))
    db_est = db(rms(model['est_err']))
    mean = db_val.mean(1)
    std = db_val.std(1)
    best = db_val.min(1)
    idx = np.argmin(db_val, axis=1)
    d = {'descrip': model['descrip'], 'db':db_val, 'mean': mean, 'std': std,
         'best': best, 'idx':idx, 'opt_path':model['opt_path'], 'db_est':db_est,
         'val_err':model['val_err'], 'models':model['models']}
    return d

def print_model(model):
    # print parameters in org-table. np.printoptions is not needed.
    with np.printoptions(precision=2, suppress=True):
        des = ''
        des += "| type | " + "|".join(str(x) for x in model['descrip']) + "|\n"
        des += "|-+" + "+".join('-' for x in list(model['descrip'])) + "|\n"
        des += "| mean | " + "|".join(f'{x:.2f}' for x in model['mean']) + "|\n"
        des += "| std | " + "|".join(f'{x:.2f}' for x in model['std']) + "|\n"
        des += "| best | " + "|".join(f'{x:.2f}' for x in model['best']) + "|\n"
        des += "| est | " + "|".join(f'{x:.2f}' for x in model['db_est']) + "|\n"
        des += "| Realization (0-index) | " + "|".join(f'{x:d}' for x in
                                                       model['idx']) + "|\n"
        des += "|-+" + "+".join('-' for x in list(model['descrip'])) + "|\n"
        print(des)

def merge_models(*_models):
    # merge the following parameters from several dataset into new dict
    descrip, mean, std, best, db_est, idx, models = ([],[],[],[],[],[],[])
    for model in _models:
        descrip += list(model['descrip'])
        mean.extend(model['mean'])
        std.extend(model['std'])
        best.extend(model['best'])
        db_est.extend(model['db_est'][1:])
        idx.extend(model['idx'])
        models.extend(model['models'])
    d = {}
    for i in ('descrip', 'mean', 'std', 'best', 'db_est', 'idx', 'models'):
        d[i] = locals()[i]
    return d

def extract_val_err(model, models=np.s_[:]):
    # extract relevant val error. So we don't pollute plot with unneeded data
    err= model['val_err'][models]
    idx = model['idx'][models]
    if isinstance(models, slice):
        des = model['descrip'][models]
    else:
        des = [model['descrip'][i] for i in models]
    global data
    global descrip
    descrip += des
    for i, j in enumerate(idx):
        data.append(err[i,j])

def print_modal(models):
    # print physical parameters in org table
    wn, zeta, knl = ([],[],[])
    for model in models['models']:
        modal = model.modal
        try:
            _knl = model.nl_coeff(0)[1].mean(1)
            if _knl.size == 0:
                _knl = np.zeros(2, dtype=int)
        except AttributeError:
            _knl = np.zeros(2, dtype=int)
        wn.append(modal['wn'][0])
        zeta.append(modal['zeta'][0])
        knl.append(_knl)

    knl = np.array(knl)
    des = ''
    des += "| Parameter | " + "|".join(str(x) for x in models['descrip']) + "|\n"
    des += "|-+" + "+".join('-' for x in models['descrip']) + "|\n"
    des += "| Nat freq (Hz)| " + "|".join(f'{x:.2f}' for x in wn) + "|\n"
    des += "| Damp ratio (%)| " + "|".join(f'{x*100:.2f}' for x in zeta) + "|\n"
    des += "| c_1 Real| " + "|".join(f'{round(x[0].real,2)}' for x in
                                     knl).replace('0.0','0') + "|\n"
    des += "| c_1 Imag| " + "|".join(f'{x[0].imag:0.2e}' for x in
                                     knl).replace('0.00e+00','0') + "|\n"
    des += "| c_2 Real| " + "|".join(f'{round(x[1].real,2)}' for x in
                                     knl).replace('0.0','0') + "|\n"
    des += "| c_2 Imag| " + "|".join(f'{x[1].imag:0.2e}' for x in
                                     knl).replace('0.00e+00','0') + "|\n"
    des += "|-+" + "+".join('-' for x in models['descrip']) + "|\n"
    print(des)

fnsi = load_sim('fnsi')
pnlss = load_sim('pnlss')
fnsid = calc_model(fnsi)
pnlssd = calc_model(pnlss)

model = merge_models(fnsid,pnlssd)
print_model(model)
print_modal(model)

figs = {}
plt.ion()
def opt_path(*data, descrip='', samefig=False):
    # optimization path
    if not samefig:
        plt.figure()
    data = data if isinstance(data[0], list) else [data]
    for j, errs in enumerate(data):
        for err in (errs):
            plt.plot(db(err))
            imin = np.argmin(err)
            plt.scatter(imin, db(err[imin]))
    plt.xlabel('Successful iteration number')
    plt.ylabel('Validation error [dB]')
    plt.xlim([0,50])
    plt.title(f'Selection of the best model on a separate data set: {descrip}')
    figs[f'{descrip}_path'] = (plt.gcf(), plt.gca())


opt_path([pnlssd['opt_path'][2][i] for i in [pnlssd['idx'][2]]])
opt_path([pnlssd['opt_path'][3][i] for i in [pnlssd['idx'][3]]], samefig=True)
opt_path([fnsid['opt_path'][4][i] for i in [fnsid['idx'][4]]], samefig=True)
opt_path([fnsid['opt_path'][5][i] for i in [fnsid['idx'][5]]], samefig=True)
plt.legend(('pnlss','pnlss weight','fnsi','fnsi weight'))

# extract relevant val error. So we don't pollute plot with unneeded data
data = []
descrip = []
extract_val_err(fnsid,[0,1,2,4,5])
extract_val_err(pnlssd, [2])
data = np.array(data)

yval = yval_raw[:,:,40,-1]
plt.figure()
N = len(yval)
freq = np.arange(N)/N*fs
plottime = np.hstack((yval, data.T))
plotfreq = np.fft.fft(plottime, axis=0)/np.sqrt(N)
nfd = plotfreq.shape[0]
plt.plot(freq[lines], db(plotfreq[lines]), '.')
plt.plot(freq[lines], db(noise[lines] / np.sqrt(N)), 'k.')
plt.ylim([-110,10])
plt.xlabel('Frequency')
plt.ylabel('Output (errors) (dB)')
plt.legend(['Output'] + descrip + ['Noise'])
plt.title(f'Validation results')
figs[f'val_data'] = (plt.gcf(), plt.gca())


# # result on estimation data
# resamp = 1
# plt.figure()
# plt.plot(est_err)
# plt.xlabel('Time index')
# plt.ylabel('Output (errors)')
# plt.legend(('Output',) + descrip)
# plt.title('Estimation results')
# figs['estimation_error'] = (plt.gcf(), plt.gca())


if savefig:
    for k, fig in figs.items():
        fig = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(fig):
            f[0].tight_layout()
            f[0].savefig(f"fig/silverbox_jp_{k}{i}.pdf")
