#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eigvals, inv, block_diag, eig
from scipy.sparse.csgraph import reverse_cuthill_mckee

class Hills(object):
    """Estimate Floquet multipliers from Hills method.

    If one of the Floquet exponent have a positive real part, the solution is
    unstable.
    Only the 2n eigenvalues with lowest imaginary modulus approximate the
    Floquet multipliers λ. The rest are spurious. (n is the number of DOFs)

    The Hill matrix is simply a by-product of a harmonic-balance continuation
    method in the frequency domain. Thus there is no need to switch from one
    domain to another for computing stability.
    (The only term that needs to be evaluated when z varies is h_z)
    The monodromy matrix might be faster computationally-wise, but this is a
    by-product of the shooting continuation method in the time-domain approach.
    Hills method is effective for large systems.[1]_

    Assemble Δs of the linear eigenvalue problem, eq. 44. B2 is assembled now.
    B1 requires h_z and is thus assembled after the steady state solution is
    found.

    Notes
    -----
    [1]_: Peletan, Loïc, et al. "A comparison of stability computational
    methods for periodic solution of nonlinear problems with application to
    rotordynamics." Nonlinear dynamics 72.3 (2013): 671-682.
    """

    def __init__(self, hb):

        scale_t = hb.scale_t
        scale_x = hb.scale_x
        NH = hb.NH
        M0 = hb.M * scale_t**2 / scale_x
        C0 = hb.C * scale_t / scale_x
        K0 = hb.K / scale_x

        Delta2 = M0
        M_inv = inv(M0)
        Delta2_inv = M_inv

        for i in range(1,NH+1):
            Delta2 = block_diag(Delta2, M0, M0)
            Delta2_inv = block_diag(Delta2_inv, M_inv, M_inv)

        # eq. 45
        b2 = np.vstack((
            np.hstack((Delta2, np.zeros(Delta2.shape))),
            np.hstack((np.zeros(Delta2.shape), np.eye(Delta2.shape[0]))) ))

        b2_inv = - np.vstack((
            np.hstack((Delta2_inv, np.zeros(Delta2_inv.shape))),
            np.hstack((np.zeros(Delta2_inv.shape), np.eye(Delta2.shape[0]))) ))

        self.Delta2 = Delta2
        self.b2_inv = b2_inv
        self.hb = hb
        #return Delta2, b2_inv

    def stability(self, omega, J_z, it=None):
        """Calculate B, Hills matrix.

        The 2n eigenvalues of B with lowest imaginary part, is the estimated
        Floquet exponents. They are collected in B_tilde.

        Returns
        -------
        B: ndarray (2Nh+1)2n x (2Nh+1)2n
            Hills coefficients
        """
        scale_x = self.hb.scale_x
        scale_t = self.hb.scale_t
        M0 = self.hb.M * scale_t**2 / scale_x
        C0 = self.hb.C * scale_t / scale_x
        K0 = self.hb.K / scale_x

        n = self.hb.n
        rcm_permute = self.hb.rcm_permute
        NH = self.hb.NH
        nu = self.hb.nu

        Delta2 = self.Delta2
        b2_inv = self.b2_inv

        omega2 = omega/nu
        # eq. 38
        Delta1 = C0
        for i in range(1,NH+1):
            blk = np.vstack((
                np.hstack((C0, - 2*i * omega2/scale_t * M0)),
                np.hstack((2*i * omega2/scale_t * M0, C0)) ))
            Delta1 = block_diag(Delta1, blk)

        # eq. 45
        A0 = J_z/scale_x
        A1 = Delta1
        A2 = Delta2
        b1 = np.vstack((
            np.hstack((A1, A0)),
            np.hstack((-np.eye(A0.shape[0]), np.zeros(A0.shape))) ))

        # eq. 46
        mat_B = b2_inv @ b1
        if rcm_permute:
            # permute B to get smaller bandwidth which gives faster linalg comp.
            p = reverse_cuthill_mckee(mat_B)
            B = mat_B[p]
        else:
            B = mat_B

        return B

    def reduc(self, B):
        """Extract Floquet exponents λ from Hills matrix.

        Find the 2n first eigenvalues with lowest imaginary part
        """

        n = self.hb.n
        w = eigvals(B)
        idx = np.argsort(np.abs(np.imag(w)))[:2*n]
        B_tilde = w[idx]

        if np.max(np.real(B_tilde)) <= 0:
            stab = True
        else:
            stab = False
        return B_tilde, stab

    def vec(self, B):
        """Find the 2n first eigenvalues with lowest imaginary part and right
        eigenvectors
        """

        n = self.hb.n
        # w: eigenvalues. vr: right eigenvector
        # Remember: column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        w, vr = eig(B)

        idx = np.argsort(np.abs(np.imag(w)))[:2*n]
        B_tilde = w[idx]

        return B_tilde, vr, idx
