#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import solve, lstsq, norm, eigvals

from .newmark import Newmark
from .common import modal_properties_MKC
from .helper.plotting import Anim

class NNM():
    def __init__(self, M, K, nonlin, omega_min, omega_max, step=0.1,
                 step_min=0.01, step_max=1, adaptive_stepsize=True,
                 opt_it_NR=3, max_it_NR=15, tol_NR=1e-6, scale=1,
                 max_it_cont=100, mode=0,
                 angle_max_beta=90, anim=True, loglevel=0,
                 unit='Hz',sca=1/(2*np.pi)):
        """Calculate NNM using the shooting method to calculate a periodic
        solution and then Pseudo-arclength to continue(follow) the
        solution/branch.
        """

        # Force C to zero, if user forgets it! :)
        self.M, self.C, self.K = M, np.zeros(M.shape), K
        self.nonlin = nonlin
        self.ndof = M.shape[0]

        self.X0_vec = []
        self.xamp_vec = []
        self.omega_vec = []
        self.energy_vec = []
        self.step_vec = []
        self.beta_vec = []
        self.stab_vec = []
        # Floquet multipliers, ie σ.
        self.flo_vec = []
        self.predict_vec = []

        self.h = step
        self.hmin = step_min
        self.hmax = step_max
        self.opt_it_NR = opt_it_NR
        self.max_it_NR = max_it_NR
        self.tol_NR = tol_NR
        self.betamax = angle_max_beta
        self.omega_min = omega_min
        self.omega_max = omega_max
        self.max_it_cont = max_it_cont
        self.adaptive_stepsize = adaptive_stepsize
        self.scale = scale
        self.mode = mode

        self.stability = True
        self.betamin = 0
        # modulus should be equal or less than 1+tol for stability, ie. on the
        # complex unit circle when using Floquet multipliers.
        self.tol_stability = 1e-3
        self.sensitivity = True
        self.loglevel = loglevel
        self.anim = anim
        self.sca = sca
        self.unit = unit

        # number points per period
        self.nppp = 360
        self.PhiT = []

        self.newmark = Newmark(self.M, self.C, self.K, nonlin)

    def periodic(self):
        """Find periodic solution using the shooting technique.

        """

        ndof = self.ndof
        # mode shape estimate
        sd = modal_properties_MKC(self.M, self.K)
        w0 = sd['wn'][self.mode]*2*np.pi
        X0 = sd['realmode'][self.mode]
        scale_alpha = np.sqrt(2*self.scale / (X0 @ self.K @ X0))
        # initial state
        X0 = np.append(scale_alpha*X0, np.zeros(ndof))
        if w0 == 0:
            w0 = 1e-3
        n = len(X0)
        indcont = np.arange(ndof)
        X0, w, XT, PhiT, ampl, cvg = self.shooting_periodic(X0, w0, indcont)
        XT = X0.copy()

        if cvg is False:
            raise ValueError('Continuation is impossible: non convergent at'
                             'first iteration, use a better initial guess.')
        self.PhiT = PhiT
        self.append_sol(X0, ampl, w, beta=0, predict=np.zeros(n+1), PhiT=PhiT)
        print('Frequency: {:g} {}\t\tEnergy: {:0.3e}'.
              format(w*self.sca, self.unit, self.energy_vec[-1]))

        return X0, w, PhiT

    def continuation(self):
        """Continuation of periodic solution using Pseudo-arclength.
        """

        h = self.h
        w = self.omega_vec[-1]
        PhiT = self.PhiT
        X0 = self.X0_vec[-1]
        XT = X0.copy()
        indcont = np.arange(self.ndof)
        # If periodic succeed, cvg is True
        cvg = True
        T = 2 * np.pi / w
        n = len(X0)

        if self.anim:
            par = {'title':'Frequency Energy plot (FEP)','xstr':'Log10(Energy) (J)',
                   'ystr':'Frequency ({})'.format(self.unit),
                   'yscale':self.sca}
            anim = Anim(x=np.log10(self.energy_vec), y=self.omega_vec,**par)

        if self.adaptive_stepsize:
            h = self.adaptive_h(h)
        if w < self.omega_max:
            smark = 1
            h = - abs(h)
        else:
            smark = -1
            h = abs(h)

        X0_predict = np.zeros(n)
        II = np.eye(n)
        beta = 0

        p = []
        cont = False
        it_w = 0
        while(it_w < self.max_it_cont and
              w*smark < self.omega_max*smark and w > self.omega_min):

            # Prediction step, p=[pz, pT]. Calculate the tangent p from
            # ∂H/∂z|t=T * pz + ∂H/∂T|t=T *pT = 0
            # ∂H/∂z = Φ - I, where Φ is the sensitivity/monodromy matrix and
            # ∂H/∂T = g(z) is the state space system. pT=1 (chosen)
            if cvg is True:
                A = PhiT[:,indcont] - II[:,indcont]
                fT = self.state_syst_cons(XT)
                sol, *_ = lstsq(A, fT)
                px = -sol
                if cont:
                    p_old = p
                    if self.adaptive_stepsize:
                        h = self.adaptive_h(h, it_NR, beta)
                p = np.append(px,1)
                p = h * p / norm(p)
                if cont:
                    p = np.sign(p @ p_old) * p

                X0_predict[indcont] = X0[indcont] + p[:-1]
                T = T + p[-1]
            else:
                if abs(h) <= self.hmin:
                    print('Stepsize: No convergence at the previous step. but'
                          'step=step_min already.')
                h = h / 2
                h = max(self.hmin, abs(h))
                print('Stepsize: decreased. No convergence at the previous'
                      'step.')
                p = h * p / norm(p)
                X0_predict[indcont] = self.X0_vec[-1][indcont] + p[:-1]
                T = 2 * np.pi / self.omega_vec[-1] + p[-1]

            X0 = X0_predict
            w_predict = 2*np.pi / T
            w = w_predict

            if T < 0:
                raise ValueError('The guess frequency ω should be larger than'
                                 '0', w)

            XT, PhiT, ampl = self.monodromy_sa(X0, T)
            H = norm(X0 - XT) / norm(X0)

            it_NR = 0
            print('Iter |        H        |       Ds        |')
            print(' {:3d} |    {:0.3e}    |                 |'.
                  format(it_NR, H))
            # NR correction.
            # |∂H/∂z|t=T, ∂H/∂T | * |Δz| = -|H| = -|z0 - zT|
            # | px      , pT    |   |ΔT|    |0|    |0      |
            while(it_NR <= self.max_it_NR and H > self.tol_NR):
                fT = self.state_syst_cons(XT)
                As = np.vstack((
                    np.hstack((PhiT[:,indcont] - II[:, indcont], fT[:,None])),
                    np.append(px, 1)
                ))
                bs = np.append(X0 - XT,0)
                Ds, *_ = lstsq(As, bs)
                X0[indcont] = X0[indcont] + Ds[:-1]
                T = T + Ds[-1]
                if T < 0:
                    print('The frequency ω became smaller than 0 during'
                          'NR correction')
                    break
                XT, PhiT, ampl = self.monodromy_sa(X0, T)
                err = H
                H = norm(X0 - XT) / norm(X0)

                if H > err:
                    msg = 'not convergent, error is not decreasing'
                    print('####' + msg + '####')
                    # raise ValueError(msg)

                it_NR += 1
                print(' {:3d} |    {:0.3e}    |    {:0.3e}    |'.
                      format(it_NR, H, norm(Ds)))

            w = 2*np.pi / T
            print('it: {:3d}  Step: {:0.3g}  Freq: {:3g} {}'.
                  format(it_w, h, w*self.sca, self.unit), end='')

            cvg = False
            if it_NR > self.max_it_NR:
                print('\n----- divergence -----')
            if H <= self.tol_NR:
                cvg = True
                if it_w > 1:
                    X0_prev = self.X0_vec[-1][indcont]
                    T_prev = 2*np.pi / self.omega_vec[-1]
                    v1 = np.append(X0[indcont],T) - np.append(X0_prev, T_prev)
                    v2 = p
                    beta = abs(np.arccos((v1 @ v2)/(norm(v1)*norm(v2))))*180/np.pi
                    Drel = norm(v1 / np.append(X0[indcont],T))
                    if(beta > self.betamax and Drel > 1e-3):
                        print('Angle condition is not fulfilled:'
                              'risk of branching switching.'
                              'Stepsize has to be decreased')
                        cvg = False
                else:
                    beta = 0

                if cvg:
                    cont = True
                    predict = np.append(X0_predict,w_predict)
                    self.append_sol(X0, ampl, w, beta, predict, PhiT)
                    print('  Energy: {:0.3e}  Stable: {}'.
                          format(self.energy_vec[-1], self.stab_vec[-1]))
                    print('----- convergence -----')
                if self.anim:
                    anim.update(x=np.log10(self.energy_vec), y=self.omega_vec)

            it_w += 1

    def shooting_periodic(self, x0, omega, indcont):

        T = 2*np.pi / omega
        if T <= 0:
            raise ValueError('The guess frequency ω should be larger than 0',
                             omega)

        n = len(x0)
        x0_in = x0[indcont].copy()

        xT, PhiT, ampl = self.monodromy_sa(x0, T)
        H = norm(x0 - xT) / max(1, norm(x0))
        it = 0
        II = np.eye(n)
        print('\nIter |        H        |       Ds        |')
        print(' {:3d} |    {:0.3e}    |                 |'.format(it, H))
        while(it <= self.max_it_NR and H > self.tol_NR):
            # Due to phase conditions, all velocities are set to zero which
            # result in all velocities being removed from the unknowns.
            fT = self.state_syst_cons(xT)
            A = np.vstack((
                np.hstack((PhiT[:,indcont] - II[:, indcont], fT[:,None])),
                np.append(x0_in, 0)
            ))
            b = np.append(x0 - xT,0)
            Ds, *_ = lstsq(A, b)

            x0[indcont] = x0[indcont] + Ds[:-1]
            T = T + Ds[-1]
            if T < 0:
                print('The frequency ω became smaller than 0 during the'
                      'iteration process')
                break

            xT, PhiT, ampl = self.monodromy_sa(x0, T)
            H = norm(x0 - xT) / norm(x0)
            it += 1
            print(' {:3d} |    {:0.3e}    |    {:0.3e}    |'.
                  format(it, H, norm(Ds)))

        cvg = False
        if H <= self.tol_NR:
            print('----- convergence -----')
            cvg = True
            omega = 2 * np.pi / T
        elif it > self.max_it_NR:
            print('----- divergence -----')

        return x0, omega, xT, PhiT, ampl, cvg

    def monodromy_sa(self, X0, T):

        x, xd, xdd, PhiT, _ = self.numsim(X0, T)
        XT = np.append(x[:,-1], xd[:,-1])
        ampl = np.max(abs(x),axis=1)

        # if PhiT is empty, calculate it by pertubating finite difference.
        if len(PhiT) == 0:

            emax = 1e-9
            emin = 1e-9
            err = norm(X0 - XT)
            e_j = max(abs(min(err, emax)), emin)

            n = len(X0)
            PhiT = np.empty((n,n))
            for j in range(n):
                dX0_j = np.zeros(n)
                if X0[j] != 0:
                    s = np.sign(X0[j])
                else:
                    s = 1

                dX0_j[j] = e_j * max(1e-2, abs(X0[j])) * s
                X0_j = X0 + dX0_j
                XT_j, *_ = self.numsim(X0_j, T)

                PhiT[:,j] = (XT_j - XT) / dX0_j[j]

        return XT, PhiT, ampl

    def numsim(self, X0, T):
        """Numerical integrate over one period"""

        n = len(X0)
        x0 = X0[:n//2]
        xd0 = X0[n//2:n]

        dt = T / self.nppp
        # We want one complete period.
        # ns = int(round((T-0)/dt)) + 1
        ns = self.nppp + 1
        fext = np.zeros(ns)
        if self.sensitivity:
            x, xd, xdd, Phi = self.newmark.integrate_nl(x0, xd0, dt, fext,
                                                        sensitivity=True)
        else:
            x, xd, xdd = self.newmark.integrate_nl(x0, xd0, dt, fext,
                                                   sensitivity=False)
            Phi = np.array([])

        return x, xd, xdd, Phi, dt

    def energy_fct(self, X):
        """Calculate the total energy, E = Ekin(𝓣) + Epot(𝓥)

        Ekin = 1/2 ẋMẋ
        Epot = 1/2 xKx + 𝓥nl(x)
        𝓥nl(x) is the nonlinear (potential) strain energy.
        """

        n = len(X)
        x = X[:n//2]
        xd = X[n//2:n]
        El = 0.5 * x @ self.K @ x + \
            0.5 * xd @ self.M @ xd
        Enl = self.nonlin.energy(x,xd)
        energy = El + Enl

        return energy.item()

    def state_syst_cons(self, X):
        """Return the state space formulation.

        ẋ  = xd
        ẋd = - M^(-1)*(Kx + fnl)

        This is also equal to the partial derivative of the shooting function
        H.
        ∂H/∂T = ∂z/∂t|t=T = g(z(T,z0))
        where g is the above state space function
        """
        n = len(X)
        x = X[:n//2]
        xd = X[n//2:n]
        g = np.zeros(n)
        fl = self.K @ x
        fnl = self.nonlin.force(x, xd)

        # TODO M could be factorized only once. Maybe faster for big systems
        g[:n//2] = xd
        g[n//2:n] = -solve(self.M, fl + fnl)

        return g

    def adaptive_h(self, h, it=None, beta=None):
        hmin, hmax = self.hmin, self.hmax
        h_old = h

        # Be more silent!
        if (it is not None and it != self.opt_it_NR):
            if self.loglevel == 1:
                print('Stepsize: not optimal convergence at the previous'
                      'step.', end='')
            h = h * (self.opt_it_NR + 1) / (it + 1)

        if (beta is not None and beta < self.betamin and beta != 0):
            print('Stepsize: should be increased (beta < betamin). ', end='')
            h = 10 * (self.betamin / beta) * h

        if (hmax != 0 and hmin != 0):
            h = max(min(hmax, abs(h)), hmin)
        elif hmax != 0:
            h = min(hmax, abs(h))
        elif hmin != 0:
            h = max(hmin, abs(h))

        if self.loglevel == 1:
            if h < h_old:
                print('Stepsize decreased.')
            elif h > h_old:
                print('Stepsize increased.')

        return h

    def append_sol(self, X0, ampl, w, beta, predict, PhiT):
        energy = self.energy_fct(X0)
        self.X0_vec.append(X0.copy())
        self.xamp_vec.append(ampl)
        self.omega_vec.append(w)
        self.energy_vec.append(energy)
        self.beta_vec.append(beta)
        self.predict_vec.append(predict)
        if self.stability:
            flo = eigvals(PhiT)
            self.flo_vec.append(flo)
            # modulus should be equal or less than 1+tol for stability, ie. on
            # the complex unit circle for Hamiltonian systems, where the
            # monodromy matrix is sympletic. This is the Floquet multipliers σ.
            # They are related to the exponents as σ=e^(λ*T) where T is the
            # period.
            if max(abs(flo)) <= 1 + self.tol_stability:
                self.stab_vec.append(True)
            else:
                self.stab_vec.append(False)
