#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from ..common import db
from ..lti_conversion import ss2frf

class scaler():
    def __init__(self, scale):
        self.scale = scale

    def ratio(self):
        return self.scale

    def label(self):
        if self.scale == 1:
            xstr = '(Hz)'
        else:
            xstr = '(rad/s)'
        return xstr


def fig_ax_getter(fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()
    return fig, ax

def plot_knl(fnsi, sca=1):
    fs = fnsi.signal.fs
    npp = fnsi.signal.npp
    flines = fnsi.flines

    freq = np.arange(npp)*fs/npp * sca
    freq_plot = freq[flines]

    if sca == 1:
        xstr = '(Hz)'
    else:
        xstr = '(rad/s)'

    figs = []
    axs = []
    for i, knl in enumerate(fnsi.knl):
        mu = knl

        mu_mean = np.zeros(2)
        mu_mean[0] = np.mean(np.real(mu))
        mu_mean[1] = np.mean(np.imag(mu))
        # ratio of 1, is a factor of 10. 2 is a factor of 100, etc
        ratio = np.log10(np.abs(mu_mean[0]/mu_mean[1]))
        exponent = 'x'.join(str(x) for x in fnsi.xpowers[i])
        print('exp: {:s}\n ℝ(mu) {:e}\n 𝕀(mu)  {:e}'.format(exponent, *mu_mean))
        print(f' Ratio log₁₀(ℝ(mu)/𝕀(mu))= {ratio:0.2f}')

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
        ax1.set_title(f'Exponent: {exponent}. Estimated: {mu_mean[0]:0.3e}')
        ax1.plot(freq_plot, np.real(mu),label='fnsi')
        ax1.axhline(mu_mean[0], c='k', ls='--', label='mean')
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax1.set_xlabel('Frequency ' + xstr)
        ax1.legend()

        str1 = ''
        ymin = np.min(np.real(mu))
        ymax = np.max(np.real(mu))
        if np.abs(ymax - ymin) <= 1e-6:
            ymin = 0.9 * mu_mean[0]
            ymax = 1.1 * mu_mean[0]
            ax1.set_ylim([ymin, ymax])
            str1 = ' 1%'
        ax1.set_ylabel(rf'Real($\mu$) $(N/m^{{{exponent}}})${str1}')

        ax2.plot(freq_plot, np.imag(mu))
        ax2.set_title(f'Ratio log₁₀(ℝ(mu)/𝕀(mu))= {ratio:0.2f}')
        ax2.axhline(mu_mean[1], c='k', ls='--', label='mean')
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
        ax2.set_xlabel('Frequency ' + xstr)
        str1 = ''
        ymin = np.min(np.imag(mu))
        ymax = np.max(np.imag(mu))
        if np.abs(ymax - ymin) <= 1e-6:
            ymin = 0.9 * mu_mean[1]
            ymax = 1.1 * mu_mean[1]
            ax2.set_ylim([ymin, ymax])
            str1 = ' 1%'
        ax2.set_ylabel(rf'Imag($\mu$) $(N/m^{{{exponent}}})${str1}')
        fig.tight_layout()
        figs.append(fig)
        axs.append([ax1, ax2])

    return figs, axs

def plot_modes(idof, sr, sca=1, fig=None, ax=None, **kwargs):
    fig, ax = fig_ax_getter(fig, ax)
    if sca == 1:
        xstr = '(Hz)'
    else:
        xstr = '(rad/s)'

    ax.set_title('Linear modes')
    ax.set_xlabel('Node id')
    ax.set_ylabel('Displacement (m)')
    # display max 8 modes
    nmodes = min(len(sr['wd']), 8)
    for i in range(nmodes):
        natfreq = sr['wn'][i]
        ax.plot(idof, sr['realmode'][i],'-*', label='{:0.2f} {:s}'.
                format(natfreq*sca, xstr))
    ax.axhline(y=0, ls='--', lw='0.5',color='k')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.legend()

    return fig, ax

def plot_frf(freq, G, p=0, m=0, sca=1, fig=None, ax=None, *args, **kwargs):
    """FRF plot of specified input/output from frequency response matrix

    Parameters
    ----------
    G : ndarray(F,p,m)
        frequency response matrix (FRM)
    freq : ndarray(F)
        vector of frequencies at which the FRM is given (in Hz)
    """
    fig, ax = fig_ax_getter(fig, ax)

    m = np.atleast_1d(m)
    p = np.atleast_1d(p)

    for i in p:
        for j in m:
            ax.plot(freq*sca, db(np.abs(G[:,i,j])), *args, **kwargs)

    if ax is None:
        ax.set_title('Nonparametric linear FRF')
    if sca == 1:
        xstr = '(Hz)'
    else:
        xstr = '(rad/s)'
    ax.set_xlabel('Frequency ' + xstr)
    ax.set_title(r'$G_{{{}{}}}$'.format(i,j))

    # For linear scale: 'Amplitude (m/N)'
    ax.set_ylabel('Amplitude (dB)')
    return fig, ax

def plot_subspace_info(infodict, fig=None, ax=None, *args, **kwargs):
    """Plot summary of subspace identification normalized by len(flines)"""
    fig, ax = fig_ax_getter(fig, ax)
    for k,v in infodict.items():
        r = np.fromiter(v.keys(), dtype=int)
        cost_sub = np.asarray([x['cost_sub'] for x in v.values()])
        stable_sub = np.asarray([x['stable_sub'] for x in v.values()])
        cost = np.asarray([x['cost'] for x in v.values()])
        stable = np.asarray([x['stable'] for x in v.values()])

        color = next(ax._get_lines.prop_cycler)['color']
        ax.plot(r, cost_sub, '.', color=color, label=f"n: {k}")
        ax.plot(r[~stable_sub], cost[~stable_sub], 'o', mfc='none', color='k')
        ax.plot(r, cost, '*', mfc='none', color=color)
        ax.plot(r[~stable], cost[~stable], 'o', mfc='none', color='k')

    ax.set_title('Cost functions of\n'
                 'subspace models (dots, stabilized models encircled in black)\n'
                 'LM-optimized models (stars, unstable models encircled in black)')
    ax.set_xlabel('r')
    ax.set_ylabel(r'$Normalized V_{WLS}$')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='center right')

    return fig, ax

def plot_subspace_model(models, G, covG, norm_freq, fs, *args, **kwargs):
    """Plot identified subspace models"""
    dictget = lambda d, *k: [d[i] for i in k]
    F, m, p = G.shape

    stdG = None
    if covG is not None and len(covG) > 1:
        tmp = np.empty((F,m*p), dtype=complex)
        for i in range(m*p):
            tmp[:,i] = covG[:,i,i]
        tmp2 = np.empty_like(G, dtype=complex)
        for f in range(F):
            tmp2[f] = tmp[f].reshape((p,m))
        stdG = np.sqrt(tmp2)

    #len(models)
    figs = []
    for k, model in models.items():
        fig, ax = plt.subplots(nrows=1, ncols=1)
        A, B, C, D = dictget(model, 'A', 'B', 'C', 'D')
        Gss = ss2frf(A,B,C,D,norm_freq)

        lsopt = {'ls':'none', 'marker':'.', 'mfc':'none'}
        figopt = {'fig':fig, 'ax':ax}
        # The CN notation allows to get the Nth color of the color cycle
        plot_frf(norm_freq*fs, G, **figopt, **lsopt, c='C1', label='BLA (non-par)')
        plot_frf(norm_freq*fs, Gss, **figopt, ls='-', c='C0', label='BLA (par)')
        plot_frf(norm_freq*fs, G-Gss, **figopt, **lsopt, c='r', label='error')
        if stdG is not None:
            plot_frf(norm_freq*fs, stdG, **figopt, ls='--', c='k', label='stdG')

        ax.legend(loc='upper right')
        tstr = ax.get_title() + f" | n={k}"
        ax.set_title(tstr)
        figs.append((fig,ax))

    return figs

def plot_svg(Sn, fig=None, ax=None, **kwargs):
    """Plot singular values of Sn. Alternative to stabilization diagram"""
    fig, ax = fig_ax_getter(fig, ax)
    ax.semilogy(Sn/np.sum(Sn),'sk', markersize=6)
    ax.set_xlabel('Model order')
    ax.set_ylabel('Normalized magnitude')
    return fig, ax

def plot_stab(sd,fmin=None, fmax=None, sca=1, fig=None, ax=None):
    fig, ax = fig_ax_getter(fig, ax)

    orUNS = []      # Unstabilised model order
    freqUNS = []    # Unstabilised frequency for current model order
    orSfreq = []    # stabilized model order, frequency
    freqS = []      # stabilized frequency for current model order
    orSep = []      # stabilized model order, damping
    freqSep = []    # stabilized damping for current model order
    orSmode = []    # stabilized model order, mode
    freqSmode = []  # stabilized damping for current model order
    orSfull = []    # full stabilized
    freqSfull = []
    for k, v in sd.items():
        # Short notation for the explicit for-loop
        # values = zip(v.values())
        # for freq, ep, mode, stab in zip(*values):
        for freq, ep, mode, stab in zip(v['freq'], v['zeta'],
                                        v['mode'], v['stab']):
            freq = freq*sca
            if stab:
                if ep and mode:
                    orSfull.append(k)
                    freqSfull.append(freq)
                elif ep:
                    orSep.append(k)
                    freqSep.append(k)
                elif mode:
                    orSmode.append(k)
                    freqSmode.append(freq)
                else:
                    orSfreq.append(k)
                    freqS.append(freq)
            else:
                orUNS.append(k)
                freqUNS.append(freq)

    # Avoid showing the labels of empty plots
    if len(freqUNS) != 0:
        ax.plot(freqUNS, orUNS, 'xr', ms=7, label='Unstabilized')
    if len(freqS) != 0:
        ax.plot(freqS, orSfreq, '*k', ms=7,
                label='Stabilized in natural frequency')
    if len(freqSep) != 0:
        ax.plot(freqSep, orSep, 'sk', ms=7, mfc='none',
                label='Extra stabilized in damping ratio')
    if len(freqSmode) != 0:
        ax.plot(freqSmode, orSmode, 'ok', ms=7, mfc='none',
                label='Extra stabilized in MAC')
    if len(freqSfull) != 0:
        ax.plot(freqSfull, orSfull, '^k', ms=7, mfc='none',
                label='Full stabilization')

    if fmin is not None:
        ax.set_xlim(left=fmin*sca)
    if fmax is not None:
        ax.set_xlim(right=fmax*sca)

    nvec = list(sd.keys())
    ax.set_ylim([nvec[0]-2, nvec[-1]])
    step = round(nvec[-2]/5)
    major_ticks = np.arange(0, nvec[-2]+1, step)
    ax.set_yticks(major_ticks)

    if sca == 1:
        xstr = '(Hz)'
    else:
        xstr = '(rad/s)'
    ax.set_xlabel('Frequency ' + xstr)
    ax.set_ylabel('Model order')
    ax.set_title('Stabilization diagram')
    ax.legend(loc='lower right')

    return fig, ax
