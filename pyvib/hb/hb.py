#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import block_diag, kron, solve, lstsq, norm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr as splsmr

from ..helper.plotting import Anim
from ..forcing import sineForce, toMDOF
from .hbcommon import fft_coeff, ifft_coeff, hb_signal, hb_components
from .stability import Hills
from .bifurcation import Fold, NS, BP

class HB():
    def __init__(self, M0, C0, K0, nonlin,
                 NH=3, npow2=8, nu=1, scale_x=1, scale_t=1,
                 amp0=1e-3, tol_NR=1e-6, max_it_NR=15,
                 stability=True, rcm_permute=False, anim=True,
                 xstr='Hz',sca=1/(2*np.pi)):
        """Because frequency(in rad/s) and amplitude have different orders of
        magnitude, time and displacement have to be rescaled to avoid
        ill-conditioning.

        Parameters
        ----------
        Harmonic balance parameters:
        NH: number of harmonic retained in the Fourier series
        npow: number of time samples in the Fourier transform, ie. 2^8=256
              samples
        nu: accounts for subharmonics of excitation freq w0
        amp0: amplitude of first guess

        """

        self.NH = NH
        self.npow2 = npow2
        self.nu = nu
        self.scale_x = scale_x
        self.scale_t = scale_t
        self.amp0 = amp0
        self.stability = stability
        self.tol_NR = tol_NR * scale_x
        self.max_it_NR = max_it_NR
        self.rcm_permute = rcm_permute
        self.nt = 2**npow2

        self.M = M0 * scale_x / scale_t**2
        self.C = C0 * scale_x / scale_t
        self.K = K0 * scale_x
        self.n = M0.shape[0]

        self.nonlin = nonlin
        self.anim = anim
        self.xstr = xstr
        self.sca = sca

        # number of unknowns in Z-vector, eq (4)
        self.nz = self.n * (2 * NH + 1)

        self.z_vec = []
        self.xamp_vec = []
        self.omega_vec = []
        self.step_vec = [0]
        self.stab_vec = []
        #  Floquet exponents(λ). Estimated from Hills matrix. Related to
        #  multipliers(σ) by σ=e^(λ*T) where T is the period.
        self.lamb_vec = []

        # list of bifurcations that are searched for
        self.bif = []

    def periodic(self, f0, f_amp, fdofs):
        """Find periodic solution

        NR iteration to solve:
        # Solve h(z,ω)=A(ω)-b(z)=0 (ie find z that is root of this eq), eq. (21)
        # NR solution: (h_z is the derivative of h wrt z)
        # zⁱ⁺¹ = zⁱ - h(zⁱ,ω)/h_z(zⁱ,ω)
        # h_z = A(ω) - b_z(z) = A - Γ⁺ ∂f/∂x Γ
        # where the last exp. is in time domain. df/dx is thus available
        # analytical. See eq (30)

        Parameters
        ----------
        f0: float
            Forcing frequency in Hz
        f_amp: float
            Forcing amplitude

        """
        self.f0 = f0
        self.f_amp = f_amp
        self.fdofs = fdofs

        nu = self.nu
        NH = self.NH
        n = self.n
        nz = self.nz
        scale_x = self.scale_x
        scale_t = self.scale_t
        amp0 = self.amp0
        stability = self.stability
        tol_NR = self.tol_NR
        max_it_NR = self.max_it_NR

        w0 = f0 * 2*np.pi
        omega = w0*scale_t
        omega2 = omega / nu

        t = self.assemblet(omega2)
        nt = len(t)
        u, _ = sineForce(f_amp, omega=omega, t=t)
        force = toMDOF(u, n, fdofs)

        # Assemble A, describing the linear dynamics. eq (20)
        A = self.assembleA(omega2)

        # Form Q(t), eq (8). Q is orthogonal trigonometric basis(harmonic
        # terms) Then use Q to from the kron product Q(t) ⊗ Iₙ, eq (6)
        mat_func_form = np.empty((n*nt, nz))
        Q = np.empty((NH*2+1,1))
        for i in range(nt):
            Q[0] = 1
            for ii in range(1,NH+1):
                Q[ii*2-1] = np.sin(omega * t[i]*ii)
                Q[ii*2] = np.cos(omega * t[i]*ii)

            # Stack the kron prod, so each block row is for time(i)
            mat_func_form[i*n:(i+1)*n,:] = kron(Q.T, np.eye(n))
        self.mat_func_form_sparse = csr_matrix(mat_func_form)

        if stability:
            hills = Hills(self)
            self.hills = hills

        # Initial guess for x. Here calculated by broadcasting. np.outer could
        # be used instead to calculate the outer product
        amp = np.ones(n)*amp0
        x_guess = amp[:,None] * np.sin(omega * t)/scale_x

        # Initial guess for z, z = Γ⁺x as given by eq (26). Γ is found as
        # Q(t)⊗Iₙ, eq 6
        # TODO use mat_func_form_sparse
        z_guess, *_ = lstsq(mat_func_form, x_guess.ravel())

        # Solve h(z,ω)=A(ω)-b(z)=0 (ie find z-root), eq. (21)
        print('Newton-Raphson iterative solution')

        Obj = 1
        it_NR = 1
        z = z_guess
        while (Obj > tol_NR) and (it_NR <= max_it_NR):
            H = self.state_sys(z, A, force)
            H_z = self.hjac(z, A)

            zsol, *_ = lstsq(H_z,H)
            z = z - zsol

            Obj = norm(H) / norm(z)
            print('It. {} - Convergence test: {:e} ({:e})'.
                  format(it_NR, Obj, tol_NR))
            it_NR = it_NR + 1

        if it_NR > max_it_NR:
            raise ValueError("Number of NR iterations exceeded {}."
                             " Change Harmonic Balance parameters".
                             format(max_it_NR))

        if stability:
            B = hills.stability(omega, H_z)
            B_tilde, stab = hills.reduc(B)
            self.stab_vec.append(stab)
            self.lamb_vec.append(B_tilde)

        # amplitude of hb components
        c, cnorm, phi = hb_components(scale_x*z, n, NH)
        x = hb_signal(omega, t, c, phi)
        xamp = np.max(x, axis=1)

        self.z_vec.append(z)
        self.xamp_vec.append(xamp)
        self.omega_vec.append(omega)

        if stability:
            return omega, z, stab, B
        else:
            return omega, z

    def continuation(self, omega_cont_min, omega_cont_max,
                     step=0.01,step_min=0.01, step_max=1, cont_dir=1,
                     opt_it_NR=3, it_cont_max=1e4, adaptive_stepsize=True,
                     angle_max_pred=90, dof=0,
                     detect={'fold':False,'NS':False,'BP':False},
                     default_bp=True):
        """ Do continuation of periodic solution.

        Based on tangent prediction and Moore-Penrose correction.
        """
        self.detect = detect
        self.cont_dir = cont_dir
        self.dof = dof

        z = self.z_vec[-1]
        omega = self.omega_vec[-1]

        scale_x = self.scale_x
        scale_t = self.scale_t
        nu = self.nu
        n = self.n
        f_amp = self.f_amp
        fdofs = self.fdofs

        tol_NR = self.tol_NR
        max_it_NR = self.max_it_NR
        NH = self.NH
        nz = self.nz
        stability = self.stability

        step_min = step_min * scale_t
        step_max = step_max * scale_t
        angle_max_pred = angle_max_pred*np.pi/180

        print('\n-------------------------------------------')
        print('|  Continuation of the periodic solution  |')
        print('-------------------------------------------\n')

        if stability:

            q0_BP = np.ones(nz+1)
            p0_BP = np.ones(nz+1)

            nldofs = self.nonlin.nldofs()
            # index of nonlinear dofs/Fourier coefficients
            nldofs_ext = np.kron(nldofs[:,None], np.ones(2*NH+1)) + \
                         np.kron(np.ones((len(nldofs),1)), np.arange(2*NH+1)*n)

            for k in detect.keys():
                if k == 'fold' and detect[k] is True:
                    fold = Fold(self, fdofs, nldofs_ext, tol_sec=tol_NR)
                    self.bif.append(fold)
                if k == 'NS' and detect[k] is True:
                    ns = NS(self, fdofs, nldofs_ext, tol_sec=tol_NR)
                    self.bif.append(ns)
                if k == 'BP' and detect[k] is True:
                    bp = BP(default_bp, self, fdofs, nldofs_ext, tol_sec=tol_NR)
                    self.bif.append(bp)

            tangent_LP = [0]
            tangent_BP = [0]
            tt_ns = np.zeros(2)

        anim = None
        if self.anim:
            par = {'title':'Nonlinear FRF',
                   'xstr':'Frequency ({})'.format(self.xstr),
                   'ystr':'Amplitude (m)','xscale':1/(scale_t)*self.sca,
                   'dof':dof,'ymin':0,
                   'xmin':omega_cont_min*self.sca,
                   'xmax':omega_cont_max*self.sca*1.1,
                   'hb':self,
                   }
            anim = Anim(x=self.omega_vec,
                        y=np.asarray(self.xamp_vec).T[dof],**par)

        omega2 = omega/nu
        # samme A som fra periodic calculation
        A = self.assembleA(omega2)

        it_cont = 1
        z_cont = z.copy()
        omega_cont = omega
        point = 0
        point_prev = 0
        point_pprev = 0
        index_BP = 0
        index_LP = 0
        index_NS = 0

        while(it_cont <= it_cont_max and
              omega / scale_t <= omega_cont_max and
              omega / scale_t >= omega_cont_min):

            print('[Iteration]:    {}'.format(it_cont))

            # Assemble A from eq. 31
            if it_cont == 1:
                J_z = self.hjac(z, A)
                J_w = self.hjac_omega(omega, z)
                A_pred = np.vstack((
                    np.hstack((J_z, J_w[:,None])),
                    np.ones(nz+1)))
                tangent = solve(A_pred, np.append(np.zeros(nz),1))
                tangent = tangent/norm(tangent)
            # With Moore-Penrose corrections, it is not needed to explicit
            # calculate the tangent again, since the corrections also correct
            # the tangent.

            if detect['BP'] and bp.branch_switch:
                tangent = bp.branch_dir(omega, z, tangent, xamp[dof], anim)

            tangent_pred = cont_dir * step * tangent

            z = z_cont + tangent_pred[:nz]
            omega = omega_cont + tangent_pred[nz]
            point = np.append(z,omega)
            if (it_cont >= 4 and
                ((point - point_prev) @ (point_prev - point_pprev) /
                 (norm(point - point_prev) * norm(point_prev-point_pprev)) <
                 np.cos(angle_max_pred)) and
                (it_cont > index_LP+1) and
                (it_cont > index_BP+2)):

                tangent_pred = -tangent_pred
                z = z_cont + tangent_pred[:nz]
                omega = omega_cont + tangent_pred[nz]
                point = np.append(z,omega)
                print('| Max angle reached. Predictor tangent is reversed. |')



            ## Corrector step. NR iterations
            # Follow eqs. 31-34
            omega2 = omega/nu
            t = self.assemblet(omega2)
            A = self.assembleA(omega2)
            u, _ = sineForce(f_amp, omega=omega, t=t)
            force = toMDOF(u, n, fdofs)

            it_NR = 1
            V = tangent
            H = self.state_sys(z, A, force)
            Obj = norm(H) / norm(z)

            # More-penrose updating
            while (Obj > tol_NR) and (it_NR <= max_it_NR):
                H = self.state_sys(z, A, force)
                H = np.append(H,0)
                J_z = self.hjac(z, A)
                J_w = self.hjac_omega(omega, z)
                Hx = np.vstack((
                    np.hstack((J_z, J_w[:,None])),
                    V))
                R = np.append(np.hstack((J_z, J_w[:,None])) @ V, 0)

                dNR = solve(Hx, H)
                dV = solve(Hx,R)
                #lu, piv = linalg.lu_factor(Hx)
                #dNR = linalg.lu_solve((lu, piv), H)
                #dV = linalg.lu_solve((lu, piv), R)

                dz = dNR[:nz]
                domega = dNR[nz]
                z = z - dz
                omega = omega - domega
                omega2 = omega/nu
                V = (V-dV) / norm(V-dV)
                t = self.assemblet(omega2)
                u, _ = sineForce(f_amp, omega=omega, t=t)
                force = toMDOF(u, n, fdofs)

                A = self.assembleA(omega2)
                H = self.state_sys(z, A, force)
                Obj = norm(H) / norm(z)
                # print('It. {} - Convergence test: {:e} ({:e})'.format(it_NR, Obj, tol_NR))
                it_NR += 1

            if it_NR > max_it_NR:
                z = z_cont
                omega = omega_cont
                step = step / 10
                print('The maximum number of iterations is reached without'
                      ' convergence. Step size decreased by factor 10:'
                      ' {:0.2g}'.format(step))
                continue
            tangent = V
            point = np.append(z,omega)
            if stability:
                if it_NR == 1:
                    J_z = self.hjac(z, A)
                B = self.hills.stability(omega, J_z)
                B_tilde, stab = self.hills.reduc(B)
                self.lamb_vec.append(B_tilde)
                self.stab_vec.append(stab)
                if detect['NS'] is True:
                    # NS/PD: if there's any eigenvalues of B⊙ with only
                    # imaginary part
                    tt_ns[0] = tt_ns[1]
                    tt_ns[1] = ns.identify(B_tilde)

            if stability and it_cont > 1 and it_cont > index_BP + 2:
                tangent_LP.append(tangent[-1])

                # Fold bifurcation is detected when the tangent prediction for
                # omega changes sign
                if (detect['fold'] and it_cont > fold.idx[-1] + 1 and
                    tangent_LP[-1] * tangent_LP[-2] < 0):
                    omega, z = fold.detect(omega, z, A, J_z, force, it_cont)
                    index_LP = fold.idx[-1]

                # NS bifurcation is detected when a pair of Floquet exponents
                # crosses the imaginary axis as complex conjugate
                if (detect['NS'] and it_cont > ns.idx[-1] + 1 and
                    tt_ns[1] != tt_ns[0]):
                    omega, z = ns.detect(omega, z, A, J_z, force, it_cont)
                    index_NS = ns.idx[-1]

                if (detect['BP'] and it_cont > bp.idx[-1] + 1):
                    G = np.vstack((
                        np.hstack((J_z, J_w[:,None])),
                        tangent))
                    t_BP, p0_BP, q0_BP = bp.test_func(G, p0_BP, q0_BP)
                    tangent_BP.append(np.real(t_BP))

                    if (tangent_BP[-1] * tangent_BP[-2] < 0):
                        omega, z = bp.detect(omega, z, A, J_z, J_w, tangent,
                                             force, it_cont)
                        index_BP = bp.idx[-1]


            z_cont = z.copy()
            omega_cont = omega
            point_pprev = point_prev
            point_prev = point
            c, phi, _ = hb_components(scale_x*z, n, NH)
            x = hb_signal(omega, t, c, phi)
            xamp = np.max(x, axis=1)
            self.xamp_vec.append(xamp)
            self.omega_vec.append(omega)
            self.step_vec.append(step)
            self.z_vec.append(z)

            print(' NR: {}\tFreq: {:0.3g}{}. Amp: {:0.3e}. Step: {:0.2g}. Stable: {}'
                  .format(it_NR-1, omega/scale_t*self.sca, self.xstr,
                          xamp[dof], step, stab))

            if adaptive_stepsize:
                step = step * opt_it_NR/it_NR
                step = min(step_max, step)
                step = max(step_min, step)

            if self.anim:
                anim.update(x=self.omega_vec,
                            y=np.asarray(self.xamp_vec).T[dof])
            it_cont += 1

    def state_sys(self, z, A, force):
        """Calculate the system matrix h(z,ω) = A(ω)z - b(z), given by eq. (21).

        b is calculated from an alternating Frequency/Time domain (AFT)
        technique, eq. (23).

            not known
          z +--------> b(z)
          +            ^
          | FFT⁻¹      | FFT
          v            +
         x(t)+---->f(x,ẋ,ω,t)

        The Fourier coefficients of the nonlinear forces b(z) are found

        """

        n = self.n
        nt = self.nt
        NH = self.NH
        scale_x = self.scale_x

        x = ifft_coeff(z, n, nt, NH)
        xd_dummy = 0
        fnl = self.nonlin.force(x*scale_x, xd_dummy)

        b = fft_coeff(force - fnl, NH)
        # eq. 21
        H = A @ z - b
        return H

    def hjac(self, z, A, force=None):
        """Computes the jacobian matrix of h wrt Fourier coefficients z,
        denoted h_z.

        From eq. (30)
        h_z = ∂h/∂z = A - ∂b/∂z = A - Γ⁺ ∂f/∂x Γ
        ∂f/∂x is available analytical in time domain

        It is only necessary to calculate the jacobian ∂b/∂z for nonlinear
        forces (ie. it is zero for linear forces).

        """
        nz = self.nz
        nt = self.nt
        NH = self.NH
        n = self.n
        scale_x = self.scale_x
        mat_func_form_sparse = self.mat_func_form_sparse

        if len(self.nonlin.nls) == 0:
            return A
        else:
            x = ifft_coeff(z, n, nt, NH)
            xd_dummy = 0
            dFnl_dx_tot_mat = self.nonlin.dforce(x*scale_x, xd_dummy) * scale_x
            # the derivative ∂b/∂z
            bjac = np.empty((nz, nz))
            full_rhs = dFnl_dx_tot_mat @ mat_func_form_sparse

            # TODO: mat_func_form is not square. But some factorization would
            # be nice
            for j in range(nz):
                rhs = np.squeeze(np.asarray(full_rhs[:,j].todense()))
                sol = splsmr(mat_func_form_sparse, rhs)
                x = - sol[0]
                bjac[:,j] = x

            hjac = A - bjac
            return hjac

    def hjac_omega(self, omega, z):
        """ Calculate the derivative of h wrt ω, h_ω.

        From eq. 32,
        h_ω = ∂A/∂ω*z
        """
        M = self.M
        C = self.C
        K = self.K
        NH = self.NH
        A = np.zeros(K.shape)
        for i in range(1,NH+1):
            blk = np.vstack((
                np.hstack((-2*i**2 * omega * M, -i * C)),
                np.hstack((i * C, -2*i**2 * omega * M))))
            A = block_diag(A, blk)

        return A @ z

    def assembleA(self, omega2):
        """Assemble A, describing the linear dynamics. eq (20)
        """
        M = self.M
        C = self.C
        K = self.K
        NH = self.NH

        A = K
        for i in range(1,NH+1):
            blk = np.vstack((
                np.hstack((K - (i * omega2)**2 * M, -i * omega2 * C)),
                np.hstack((i * omega2 * C, K - (i * omega2)**2 * M))))
            A = block_diag(A, blk)
        return A

    def assemblet(self, omega2):
        npow2 = self.npow2
        return np.arange(2**npow2) / 2**npow2 * 2*np.pi / omega2

    def get_components(self, omega=None, z=None):
        scale_x = self.scale_x
        scale_t = self.scale_t
        n = self.n
        NH = self.NH
        nu = self.nu

        if omega is None:
            omega = self.omega_vec[-1]
            z = self.z_vec[-1]

        c, phi, cnorm = hb_components(scale_x*z, n, NH)

        omega2 = omega/nu
        iw = np.arange(NH+1) * omega2
        cd = c*iw
        cd[:,0] = 0
        phid = phi + np.pi/2

        cdd = -c * iw**2
        cdd[:,0] = 0
        phidd = phi

        t = self.assemblet(omega2)*scale_t
        # make t one full period + dt, ie include first point of next period
        t = np.append(t, t[-1] + t[1] - t[0])
        omegap = omega/scale_t
        omega2p = omega2/scale_t
        zp = z*scale_x

        return t, omegap, zp, cnorm, (c, phi), (cd, phid), (cdd, phidd)
