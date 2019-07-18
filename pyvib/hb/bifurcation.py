#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import solve, svd, norm, lstsq, inv
from ..forcing import sineForce, toMDOF

class Bifurcation(object):
    def __init__(self, hb, fdofs, nldofs_ext, marker, stype, max_it_secant=10,
                 tol_sec=1e-12):
        """Localize the bifurcation point by solving the augmented state system
        h_aug = [h, g]^T, where g is the test function for the given
        bifurcation type."""

        self.max_it_secant = max_it_secant
        self.tol_sec = tol_sec

        self.hb = hb
        self.fdofs = fdofs
        self.nldofs_ext = nldofs_ext
        self.idx = [0]
        self.nbif = 0
        self.isbif = False
        # add description of this type of bifurcation. For plotting
        self.marker = marker
        self.stype = stype

class Fold(Bifurcation):
    def __init__(self,*args, **kwargs):
        kwargs.update({'marker':'s', 'stype':'fold'})
        super().__init__(*args, **kwargs)

    def detect(self, omega, z, A, J_z, force, it_cont):
        """Fold bifurcation traces the lotus of the frequency response peaks

        At a fold bifurcation these four conditions are true:
        * h(z,ω) = 0
        * Rank h_z(z,ω) = nz-1
        * h_ω(z,ω) ∉ range h_z(z,ω). Ie:  Rank [h_z,h_ω] = nz
        * There is a parametrization z(σ) and ω(σ), with z(σ₀) = z₀,
          ω(σ₀) = z₀ and d²ω(σ)/d²σ ≠ 0

        Ie fold bifurcations are detected when h_z is singular and Rank
        [h_z,h_ω] = nz. It is however more efficient to detect fold
        bifurcations, when the component of the tangent vector related to ω
        changes sign.
        """
        nz = self.hb.nz
        max_it_secant = self.max_it_secant
        tol_sec = self.tol_sec
        nu = self.hb.nu
        scale_x = self.hb.scale_x
        n = self.hb.n
        f_amp = self.hb.f_amp
        omega_save = omega.copy()
        z_save = z.copy()
        print('-----> Detecting LP bifurcation...')

        A0 = J_z/scale_x
        G = A0
        B = A0
        Gq0 = null_approx(G, 'LP2')
        Gp0 = null_approx(G.T, 'LP2')
        if Gq0 is False or Gp0 is False:
            return omega, z
        h = self.hb.state_sys(z, A, force)
        F, M, w = extended_state_sys(h, G, Gp0, Gq0, 'LP2')
        print('Test function = {}'.format(norm(F) / norm(z)))

        it = 0
        while (it < max_it_secant and norm(F)/norm(z) > tol_sec):

            J_ext = extended_jacobian(self.hb, omega, z, A, B, M, w,
                                      self.fdofs, self.nldofs_ext, 'LP2')

            dNR, *_ = lstsq(J_ext[:,:nz+1], -F)
            z = z + dNR[:nz]
            omega = omega + dNR[nz]

            omega2 = omega/nu
            t = self.hb.assemblet(omega2)
            u, _ = sineForce(f_amp, omega=omega, t=t)
            force = toMDOF(u, n, self.fdofs)
            A = self.hb.assembleA(omega2)

            A0 = self.hb.hjac(z, A) / scale_x
            G = A0
            B = A0
            Gq0 = null_approx(G, 'LP2')
            Gp0 = null_approx(G.T, 'LP2')
            if Gq0 is False or Gp0 is False:
                it = max_it_secant + 1; break
            h = self.hb.state_sys(z, A, force)

            F, M, w = extended_state_sys(h, G, Gp0, Gq0, 'LP2')
            print('Test function = {}'.format(norm(F)))
            it += 1

        if it >= max_it_secant:
            print('-----> LP bifurcation detection failed')
            self.isbif = False
            return omega_save, z_save
        else:
            print('-----> LP detected.')
            self.idx.append(it_cont)
            self.nbif += 1
            self.isbif = True
            return omega, z

class NS(Bifurcation):
    def __init__(self,*args, **kwargs):
        kwargs.update({'marker':'d', 'stype':'NS'})
        super().__init__(*args, **kwargs)

    def detect(self, omega, z, A, J_z, force, it_cont):
        """Detect Neimark-Sacker(NS) bifurcation.

        At a NS bifurcation(also called torus- and Hopf bifurcation), quasi
        periodic oscillation emanates. Quasi periodic oscillations contains the
        forcing frequency ω and at least one other frequency ω₂(the envelope)
        These two frequencies are incommensurate, ie ω/ω₂ is irrational.

        NS bifurcations are detected when the bialternate matrix product of B
        is singular, B_⊙=B⊙I. The bialternate matrix is singular when two of
        its eigenvalues μ₁+μ₂=0. For example: μ₁,₂=±jμ
        """
        nz = self.hb.nz
        max_it_secant = self.max_it_secant
        tol_sec = self.tol_sec
        nu = self.hb.nu
        scale_x = self.hb.scale_x
        n = self.hb.n
        f_amp = self.hb.f_amp
        omega_save = omega.copy()
        z_save = z.copy()

        print('-----> Detecting NS bifurcation...')
        B = self.hb.hills.stability(omega, J_z)
        B_tilde, vr, idx = self.hb.hills.vec(B)
        vr_inv = inv(vr)[idx,:]
        vr = vr[:,idx]

        G = bialtprod(np.diag(B_tilde))
        Gq0 = null_approx(G, 'NS')
        Gp0 = null_approx(G.T, 'NS')
        if Gq0 is False or Gp0 is False:
            return omega, z
        h = self.hb.state_sys(z, A, force)
        F, M, w = extended_state_sys(h, G, Gp0, Gq0, 'NS')
        print('Test function = {}'.format(norm(F) / norm(z)))

        it = 0
        while (it < max_it_secant and norm(F)/norm(z) > tol_sec):
            J_ext = extended_jacobian(self.hb, omega, z, A, B, M, w,
                                      self.fdofs, self.nldofs_ext, 'NS', vr,
                                      vr_inv)

            dNR, *_ = lstsq(J_ext[:,:nz+1], -F)
            z = z + dNR[:nz]
            omega = omega + dNR[nz]
            omega2 = omega/nu
            t = self.hb.assemblet(omega2)
            u, _ = sineForce(f_amp, omega=omega, t=t)
            force = toMDOF(u, n, self.fdofs)
            A = self.hb.assembleA(omega2)
            J_z = self.hb.hjac(z, A)

            B = self.hb.hills.stability(omega, J_z)
            B_tilde, vr, idx = self.hb.hills.vec(B)
            vr_inv = inv(vr)[idx,:]
            vr = vr[:,idx]

            G = bialtprod(np.diag(B_tilde))
            Gq0 = null_approx(G, 'NS')
            Gp0 = null_approx(G.T, 'NS')
            if Gq0 is False or Gp0 is False:
                it = max_it_secant + 1; break
            h = self.hb.state_sys(z, A, force)
            F, M, w = extended_state_sys(h, G, Gp0, Gq0, 'NS')
            print('Test function = {}'.format(norm(F) / norm(z)))

            it += 1
        if it >= max_it_secant:
            print('-----> NS bifurcation detection failed')
            self.isbif = False
            return omega_save, z_save
        else:
            print('-----> NS detected.')
            self.idx.append(it_cont)
            self.nbif += 1
            self.isbif = True
            return omega, z

    def identify(self, B_tilde):
        """NS is identified if there is a pair of eigenvalues with only
        an imaginary part, ie.
        μ₁ + μ₂ = 0, where
        μ₁,₂ = ±iβ
        thus when len(n_mu) = 0
        """
        G = bialtprod(np.diag(B_tilde))
        Gdiag = np.diag(G)
        Gdiag_real = Gdiag[Gdiag == Gdiag.real]
        n_mu = Gdiag_real[Gdiag_real > 0]
        return len(n_mu)

class BP(Bifurcation):
    def __init__(self, default_dir=True, *args, **kwargs):
        kwargs.update({'marker':'^', 'stype':'BP'})
        super().__init__(*args, **kwargs)
        self.branch_switch = False
        self.default_dir = default_dir

    def detect(self, omega, z, A, Jz, Jw, tangent, force, it_cont):
        """Branch point bifurcations occur when two branches of periodic
        solution meet.

        At a branch bifurcation these four conditions are true:
        * h(z,ω) = 0
        * Rank h_z(z,ω) = nz-1
        * h_ω(z,ω) ∈ range h_z(z,ω). Ie:  Rank [h_z,h_ω] = nz-1
        * Exactly two branches of periodic solutions intersect with two distinct
        tangents.
        """
        nz = self.hb.nz
        max_it_secant = self.max_it_secant
        tol_sec = self.tol_sec
        nu = self.hb.nu
        scale_x = self.hb.scale_x
        n = self.hb.n
        f_amp = self.hb.f_amp
        omega_save = omega.copy()
        z_save = z.copy()

        print('-----> Detecting BP bifurcation...')
        G = np.vstack((
            np.hstack((Jz, Jw[:,None])),
            tangent))

        Gq0 = null_approx(G,'BP')
        Gp0 = null_approx(G.T,'BP')
        if Gq0 is False or Gp0 is False:
            return omega, z
        h = self.hb.state_sys(z, A, force)
        F, M, w = extended_state_sys(h, G, Gp0, Gq0, 'BP')
        print('Test function = {}'.format(norm(F) / norm(z)))

        it = 0
        while (it < max_it_secant and norm(F)/norm(z) > tol_sec):
            J_ext = extended_jacobian(self.hb, omega, z, A, G, M, w,
                                      self.fdofs, self.nldofs_ext, 'BP',
                                      tangent=tangent)
            dNR, *_ = lstsq(J_ext[:,:nz+1], -F)
            z = z + dNR[:nz]
            omega = omega + dNR[nz]
            omega2 = omega/nu
            t = self.hb.assemblet(omega2)
            u, _ = sineForce(f_amp, omega=omega, t=t)
            force = toMDOF(u, n, self.fdofs)
            A = self.hb.assembleA(omega2)
            Jz = self.hb.hjac(z, A)
            A0 = Jz/scale_x

            Jw = self.hb.hjac_omega(omega, z)
            G = np.vstack((
                np.hstack((A0, Jw[:,None])),
                tangent))
            Gq0 = null_approx(G,'BP')
            Gp0 = null_approx(G.T,'BP')
            if Gq0 is False or Gp0 is False:
                it = max_it_secant + 1; break
            h = self.hb.state_sys(z, A, force)
            F, M, w = extended_state_sys(h, G, Gp0, Gq0, 'NS')
            print('Test function = {}'.format(norm(F) / norm(z)))

            it += 1
        if it >= max_it_secant:
            print('-----> BP bifurcation detection failed')
            self.isbif = False
            return omega_save, z_save
        else:
            print('-----> BP detected. Computing switching direction...')
            self.idx.append(it_cont)
            self.nbif += 1
            self.isbif = True

            if self.default_dir is True:
                self.branch_switch = False
            else:
                self.calc_tangent(omega, z, A, Jz, Jw, tangent)

            return omega, z

    def calc_tangent(self, omega, z, A, Jz, Jw, tangent):
        nz = self.hb.nz
        nu = self.hb.nu

        #Jz = self.hb.hjac(z, A)
        #Jw = self.hb.hjac_omega(omega, z)
        J_BP = np.vstack((
            np.hstack((Jz, Jw[:,None])),
            tangent))
        phi1, *_ = lstsq(J_BP, np.append(np.zeros(nz),1))
        phi1 = phi1 / norm(phi1)
        psi = null_approx(Jz.T,'BP')
        phi2 = null_approx(J_BP,'BP')
        beta2 = 1

        # perturbation
        # First pertubate z, then omega
        eps = 1e-8
        hess = np.empty((nz,nz+1,nz+1))
        # TODO make parallel for loop
        for i in range(nz):
            z_pert = z.copy()
            z_pert[i] = z_pert[i] + eps
            Jz_pert = self.hb.hjac(z_pert, A)
            Jw_pert = self.hb.hjac_omega(omega, z_pert)
            hess[...,i] = np.hstack(((Jz_pert - Jz)/eps,
                                     (Jw_pert - Jw)[:,None]))
        omega_pert = omega + eps
        omega2_pert = omega/nu
        A = self.hb.assembleA(omega2_pert)
        Jz_pert = self.hb.hjac(z, A)
        Jw_pert = self.hb.hjac_omega(omega_pert, z)
        hess[...,nz] = np.hstack(((Jz_pert - Jz)/eps,
                                  (Jw_pert - Jw)[:,None]))

        multi_prod12 = np.zeros(nz)
        multi_prod22 = np.zeros(nz)
        for i in range(nz):
            for j in range(nz+1):
                for k in range(nz+1):
                    multi_prod12[i] += hess[i,j,k] * phi1[j] * phi2[k]
                    multi_prod12[i] += hess[i,j,k] * phi2[j] * phi2[k]

        c12 = psi.T @ multi_prod12
        c22 = psi.T @ multi_prod22
        alpha2 = -1/2 * c22/c12 * beta2
        self.Y0_dot = alpha2*phi1 + beta2*phi2
        self.V0 = self.Y0_dot / norm(self.Y0_dot)
        self.branch_switch = True

    def test_func(self, G, p0, q0):
        return test_func(G, p0, q0, 'BP')

    def branch_dir(self, omega, z, tangent, xamp, anim):

        # new tangent direction.
        tt_bp2 = self.V0

        # plot tangent directions
        if anim is not None:
            self.plot_tangent(omega, z, tangent, xamp, anim, show=True)
        # make 1 default input
        tangent_dir = int(input("Enter tangent direction [1-3] shown by "
                                "colors [k,r,g]: ") or "1")
        if tangent_dir == 1:
            print('-----> No branch switch.')
        elif tangent_dir == 2:
            tangent = tt_bp2
            print('-----> Branch switch.')
        elif tangent_dir == 3:
            tangent = -tt_bp2
            print('-----> Branch switch.')

        # remove tangent directions from plot
        if anim is not None:
            self.plot_tangent(omega, z, tangent, xamp, anim, show=False)

        self.branch_switch = False
        return tangent

    def plot_tangent(self, omega, z, tangent, xamp, anim, show=False):
        # Only calculated for plotting, ie. show the tangent direction.
        nz = self.hb.nz
        dof = self.hb.dof

        tt_bp2 = self.V0
        omegas = [omega + tangent[nz], omega + tt_bp2[nz], omega - tt_bp2[nz]]
        zs = [z + tangent[:nz], z + tt_bp2[:nz], z - tt_bp2[:nz]]
        xmax = [self.xdisp(wp,zp,dof) for wp,zp in zip(omegas, zs)]

        w0 = omega
        anim.plot_tangent(w0, xamp, omegas, xmax, show)

    def xdisp(self, omega, z, dof):
        """Calculate the x-displacements from z and return the maximum"""

        nt = self.hb.nt
        nh = self.hb.NH
        n = self.hb.n
        freq = np.arange(nt//2) * omega/2/np.pi

        t = self.hb.assemblet(omega)
        an = np.zeros((nh+1,n))
        bn = np.zeros((nh,n))
        x = np.zeros((n,nt))
        an[0] = z[:n]
        x = np.outer(an[0][:,None], np.ones(nt))

        ind = n
        for i in range(nh):
            for j in range(n):
                an[i+1,j] = z[ind+j]
                bn[i,j] = z[ind+n+j]
            ind += 2 * n

        for i in range(1,nh+1):
            for j in range(n):
                x[j,:] += an[i,j]*np.sin(2*np.pi*t*freq[i]) + \
                             bn[i-1,j]*np.cos(2*np.pi*t*freq[i])

        # velocity
        xd = np.zeros((n,len(t)))
        for i in range(nh+1):
            for j in range(n):
                xd[j,:] += \
                    2*np.pi*freq[i]*an[i,j]*np.cos(2*np.pi*t*freq[i]) - \
                    2*np.pi*freq[i]*bn[i-1,j]*np.sin(2*np.pi*t*freq[i])

        xmax = np.max(x[dof,:])*self.hb.scale_x
        return xmax


def extended_state_sys(h, G, p0, q0, bif_type):
    """Extend the state system h with the test function g from the bordered
    system, eq. 1.66
    """

    return bordered_system(G, p0, q0, bif_type, h)

def test_func(G, p0, q0, bif_type):
    """


    Parameters
    ----------

    """
    q0 = np.zeros(q0.shape)
    q0[0] = 1
    p0 = np.zeros(p0.shape)
    p0[0] = 1

    return bordered_system(G, p0, q0, bif_type)

def bordered_system(G, p0, q0, bif_type, h=None):
    """Calculate the bordered system

    If h(the state system) is given, then return the augmented/extended state
    system. Otherwise return the test function g.


    Parameters
    ----------
    G: ndarray(matrix)
        G of the bordered system. For NS/PD, G is the bialtprod of B. As B is a
        diagonal matrix, so is G. For LP/BP: G is the Jacobian Jz.
    p0: ndarray(vec)
        nullvec of G
    q0: ndarray(vec)
        nullvec of G*
    h: ndarray(vec)
        The normal state system

    Returns
    -------
    g: scalar
        test function. Last part of solution of bordered system.
    M: ndarray(matrix)
        System matrix of bordered system
    w: ndarray(vec)
        First part of solution of bordered system
    """

    # Bordered system, eq. 1.65
    nG = G.shape[1]
    M = np.vstack((
        np.hstack((G, p0[:,None])),
        np.hstack((q0, 0))
    ))

    wg = solve(M, np.append(np.zeros(nG),1)).real
    w = wg[:nG]
    g = wg[nG]
    if h is None:
        return g, q0, p0
    h = np.append(h,g)

    return h, M, w

def null_approx(A, bif_type):
    """Compute the nullspace of A depending on the type of bifurcation"""

    if bif_type == 'LP2' or bif_type == 'BP':
        # compute nullspace from SVD decomposition
        nullvec = nullspace(A, atol=5e-1).squeeze()
        # No nullvec found. The bifurcation point is too far away.
        if nullvec.size == 0:
            return False
        # make sure there's only one vector and it is the one corresponding to
        # the lowest singular value. Maybe atol should be lower, ie. 5e-3?
        if nullvec.ndim > 1:
            nullvec = nullvec[:,-1]
        return nullvec
    else:
        if not np.all(A == np.diag(np.diagonal(A))):
            raise ValueError('The matrix A should be diagonal')

        eig_A = np.diag(A)
        idx = np.argmin(np.abs(eig_A))

        nullvec = np.zeros(A.shape[0])
        nullvec[idx] = 1

        return nullvec

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    # V: Unitary matrix having right singular vectors as rows.
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    # In case of multiple occurrences of the minimum values, the indices
    # corresponding to the first occurrence is returned.
    idx = np.argmin(np.abs(s))
    eig_sm = np.abs(s[idx])
    if eig_sm > tol:
        print('The approximation of the bifurcation point has to be improved\n'
              'The smallest eigenvalue of the matrix is {:.2e}'.format(eig_sm))

    return ns

def bialtprod(A):
    """Calculate bialternate product of A

    The bilaternate product is assembled by extracting blocks from A. The
    indexes for the extracted blocks only depends on the shape of A.
    See [1]_

    TODO: The indexing is only depending on the shape of A. Thus the found
    indexes could be saved in order to save computational time if A have the
    same shape between calls

    References
    ----------
    [1]_: Yuri A. Kuznetsov - Bialternate matrix product

    """

    n = A.shape[0]
    m = n*(n-1)//2
    B = np.zeros((m,m), dtype=A.dtype)

    Z = np.ones(m, dtype=int)
    # init indexes
    init = np.outer(np.arange(1,n), Z)
    init2 = np.outer(np.ones(n-1, dtype=int), np.arange(m))
    idx = np.where(init2 < init)
    init = init[idx]
    init2 = init2[idx]

    p = np.outer(init, Z)
    q = np.outer(init2, Z)
    r = np.outer(Z, init)
    s = np.outer(Z, init2)

    rp = r == p
    sq = s == q

    # part1
    idx = np.where(r==q)
    idx2 = _sub2ind(n, p[idx], s[idx])
    B[idx] = -A.flat[idx2]

    # part2
    idx = np.where(~rp & sq)
    idx2 = _sub2ind(n, p[idx], r[idx])
    B[idx] = A.flat[idx2]

    # part3
    idx = np.where(rp & sq)
    idx2 = _sub2ind(n, p[idx], p[idx])
    B[idx] = A.flat[idx2]
    idx2 = _sub2ind(n, q[idx], q[idx])
    B[idx] += A.flat[idx2]

    # part4
    idx = np.where(rp & ~sq)
    idx2 = _sub2ind(n, q[idx], s[idx])
    B[idx] = A.flat[idx2]

    # part5
    idx = np.where(s==p)
    idx2 = _sub2ind(n, q[idx], r[idx])
    B[idx] = -A.flat[idx2]

    return B

def _sub2ind(n, row, col):
    return row*n + col

def extended_jacobian(self, omega, z, A, B, M, w, fdofs, nldofs_ext, bif_type,
                      vr=None, vr_inv=None, tangent=None):
    """Calculate the augmented Jacobian eq. 1.67.

    J = [hz, hω, hf
         gz, gω, gf]
    where g is the test function.

    Parameters
    ----------
    M: ndarray
        System matrix in bordered system: M = [G, 0; q*, 1], eq. 1.64
    w: ndarray
        solution vector of bordered system
    fdofs: int
        Location of monoharmonic force
    nldofs_ext: ndarray
        index of nonlinear dofs/Fourier coefficients
    bif_type: str
        Bifurcation type
    vr: ndarray
        right eigenvectors of B
    vr_inv: ndarray
        Inverse of eigenvectors of B
    """
    n = self.n
    nG = M.shape[1]-1
    nz = self.nz

    Jz = self.hjac(z, A)
    Jw = self.hjac_omega(omega, z)
    # derivative of system h wrt force.
    btronc = np.zeros(nz)
    btronc[2*n+fdofs-1] = 1
    Jf = -btronc

    # transposed of bordered system. eq. 1.72
    vh = solve(M.T, np.append(np.zeros(nG),1))
    v = vh[:nG]

    # derivative of g wrt α(z, ω, f ) is, due to the bordered system,
    # gα = -v Gα w, where Gα is the derivative G wrt α. See eq. 1.70
    # Gα is calculated in hessian.
    J_part = np.zeros(nz)
    # TODO parallel loop?
    # hessian is zero for z's related to Fourier coefficients for linear DOFs.
    for i in range(nz):
        if i in nldofs_ext:
            hess = hessian(self, omega, z, A, B, i, 'z', bif_type, vr, vr_inv,
                           tangent)
            Jg_A = - v @ hess @ w
        else:
            Jg_A = 0
        J_part[i] = np.real(Jg_A)

    # TODO Rewrite so we DONT pass 0.
    hess = hessian(self, omega, z, A, B, 0,'omega', bif_type, vr, vr_inv,
                   tangent)
    Jg_p1 = - v @ hess @ w

    hess = hessian(self, omega, z, A, B, 0, 'f', bif_type, vr, vr_inv, tangent)
    Jg_p2 = - v @ hess @ w

    J_part = np.append(J_part, (np.real(Jg_p1), np.real(Jg_p2)))
    J = np.vstack((
        np.hstack((Jz, Jw[:,None], Jf[:,None])),
        J_part
    ))

    return J


def hessian(self, omega, z, A, B, idx, gtype, bif_type, vr=None, vr_inv=None,
            tangent=None):
    """Calculate hessian, ie Gα, the derivative of G wrt α(z, ω, f )

    The Hessian matrix is a square matrix of second-order partial derivatives.

    Bif type:
    LP/BP: (eq. 1.73)
        Gα = hzα
    NS/PD: (eq. 1.74)
        Gα = (∂B/∂α)⊝

    LP/BP is calculated by finite differences. NS/PD is calculated using
    properties of the derivatives of eigenvalues, eq. 1.75.

    Parameters
    ----------
    B: ndarray
        For LP2: hz (ie Jacobian). For NS/PD: B (ie. Hills full matrix)
    idx: int
        Index of z-component. 0 for ω and f.
    gtype: str
        The derivative, ie. α.
    vr: ndarray
        right eigenvectors of B
    vr_inv: ndarray
        Inverse of eigenvectors of B
    """
    scale_x = self.scale_x
    nu = self.nu

    eps = 1e-5
    if gtype == 'z':
        z_pert = z.copy()

        if z_pert[idx] != 0:
            eps = eps * abs(z_pert[idx])

        z_pert[idx] = z_pert[idx] + eps
        Jz_pert = self.hjac(z_pert, A)

        if bif_type == 'LP2':
            return (Jz_pert/scale_x - B) / eps
        elif bif_type == 'BP':
            Jw_pert = self.hjac_omega(omega, z_pert)
            dG_dalpha = (np.vstack((np.hstack((Jz_pert/scale_x, Jw_pert[:,None])),
                                   tangent)) - B) / eps
            return dG_dalpha
        elif bif_type == 'NS':
            B_pert = self.hills.stability(omega, Jz_pert)
            hess = (B_pert - B)/eps
            # derivative calculated using properties of eigenvalues, eq. 1.75
            dG_dalpha_tot = np.diag(vr_inv @ hess @ vr)
            dG_dalpha = bialtprod(np.diag(dG_dalpha_tot))
            return dG_dalpha

    elif gtype == 'omega':

        if omega != 0:
            eps = eps * abs(omega)
            omega_pert = omega + eps

        omega2_pert = omega_pert / nu
        A_pert = self.assembleA(omega2_pert)
        Jz_pert = self.hjac(z, A_pert)

        if bif_type == 'LP2':
            return (Jz_pert/scale_x - B) / eps
        elif bif_type == 'BP':
            Jw_pert = self.hjac_omega(omega_pert, z)
            dG_dalpha = (np.vstack((np.hstack((Jz_pert/scale_x, Jw_pert[:,None])),
                                   tangent)) - B) / eps
            return dG_dalpha
        elif bif_type == 'NS':
            B_pert = self.hills.stability(omega_pert, Jz_pert)
            hess = (B_pert - B)/eps
            dG_dalpha_tot = np.diag(vr_inv @ hess @ vr)
            dG_dalpha = bialtprod(np.diag(dG_dalpha_tot))
            return dG_dalpha

    elif gtype == 'f':
        if bif_type == 'LP2' or bif_type == 'BP':
            hess = np.zeros(B.shape)
        else:
            # size of bialternate product
            n = self.n*2
            m = n*(n-1)//2
            hess = np.zeros((m,m))
        return hess
