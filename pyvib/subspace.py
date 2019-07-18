#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import kron
# qr(mode='r') returns r in economic form. This is not the case for scipy
# svd and solve allows broadcasting when imported from numpy
from numpy.linalg import qr, solve, svd
from scipy.linalg import logm, lstsq, norm, pinv
from scipy.signal import dlsim

from .common import (matrix_square_inv, mmul_weight, normalize_columns,
                     weightfcn)
from .helper.modal_plotting import plot_subspace_info, plot_subspace_model
from .lti_conversion import is_stable, ss2frf
from .modal import modal_ac
from .statespace import StateSpace, StateSpaceIdent

# TODO extract_model should be refactored so the method from SS can be used
# right now it is not clear if it should be used at all

class Subspace(StateSpace, StateSpaceIdent):

    def __init__(self, signal, *system, **kwargs):
        self.signal = signal
        kwargs['dt'] = 1/signal.fs
        super().__init__(*system, **kwargs)

    @property
    def weight(self):
        if self._weight is None:
            self._weight = weightfcn(self.signal.covG)
        return self._weight

    def costfcn(self, x0=None, weight=False):
        if weight is True:
            weight = self.weight
        if x0 is None:
            x0 = self.flatten()
        return costfcn(x0, self, weight=weight)

    def jacobian(self, x0, weight=False):
        return jacobian(x0, self, weight=weight)

    def estimate(self, n, r, weight=False, copy=False):
        """Subspace estimation"""

        self.n = n
        self.r = r
        signal = self.signal
        if weight is True:
            weight = signal.covG
        A, B, C, D, z, stable = \
            subspace(signal.G, weight, signal.norm_freq, n, r)

        self.A, self.B, self.C, self.D, self.z, self.stable = \
            A, B, C, D, z, stable

        return A, B, C, D, z, stable

    def scan(self, nvec, maxr, optimize=True, method=None, weight=False,
             info=2, nmax=50, lamb=None, ftol=1e-8, xtol=1e-8, gtol=1e-8):

        F = self.signal.F
        nvec = np.atleast_1d(nvec)
        maxr = maxr
        if weight is True:
            weight = self.weight

        infodict = {}
        models = {}
        if info:
            print('Starting subspace scanning')
            print(f"n: {nvec.min()}-{nvec.max()}. r: {maxr}")
        for n in nvec:
            minr = n + 1

            cost_old = np.inf
            if isinstance(maxr, (list, np.ndarray)):
                rvec = maxr[maxr >= minr]
                if len(rvec) == 0:
                    raise ValueError(f"maxr should be > {minr}. Is {maxr}")
            else:
                rvec = range(minr, maxr+1)

            infodict[n] = {}
            for r in rvec:
                if info:
                    print(f"n:{n:3d} | r:{r:3d}")

                self.estimate(n, r)
                # normalize with frequency lines to comply with matlab pnlss
                cost_sub = self.cost(weight=weight)/F
                stable_sub = self.stable

                if optimize:
                    self.optimize(method=method, weight=weight, info=info,
                                  nmax=nmax, lamb=lamb, ftol=ftol, xtol=xtol,
                                  gtol=gtol, copy=False)

                cost = self.cost(weight=weight)/F
                stable = is_stable(self.A, domain='z')
                infodict[n][r] = {'cost_sub':cost_sub, 'stable_sub':stable_sub,
                                  'cost': cost, 'stable': stable}
                if cost < cost_old and stable:
                    # TODO instead of dict of dict, maybe use __slots__ method
                    # of class. Slots defines attributes names that are
                    # reserved for the use as attributes for the instances of
                    # the class.
                    print(f"New best r: {r}")
                    cost_old = cost
                    models[n] = {'A': self.A, 'B': self.B, 'C': self.C, 'D':
                                 self.D, 'r':r, 'cost':cost, 'stable': stable}

            self.models = models
            self.infodict = infodict
        return models, infodict

    def plot_info(self, fig=None, ax=None):
        """Plot summary of subspace identification"""
        return plot_subspace_info(self.infodict, fig, ax)

    def plot_models(self):
        """Plot identified subspace models"""
        return plot_subspace_model(self.models, self.signal.G,
                                   self.signal.covG, self.signal.norm_freq,
                                   self.signal.fs)

    def extract_model(self, y=None, u=None, models=None, n=None, t=None, x0=None):
        """extract the best model using validation data"""

        dt = 1/self.signal.fs
        if models is None:
            models = self.models

        if n is None:
            if y is None or u is None:
                raise ValueError('y and u cannot be None when several models'
                                 ' are given')
            model, err_vec = extract_model(models, y, u, dt, t, x0)
        elif {'A', 'B', 'C', 'D'} <= models.keys() and n is None:
            model = models
        else:
            model = models[n]
            err_vec = []

        dictget = lambda d, *k: [d[i] for i in k]
        self.A, self.B, self.C, self.D, self.r, self.stable = \
            dictget(model, 'A', 'B', 'C', 'D', 'r', 'stable')

        return err_vec


def jacobian_freq(A,B,C,z):
    """Compute Jacobians of the unweighted errors wrt. model parameters.

    Computes the Jacobians of the unweighted errors ``e(f) = Ĝ(f) - G(f)``
    w.r.t. the elements in the ``A``, ``B``, and ``C`` state-space matrices.
    The Jacobian w.r.t. the elements of ``D`` is a zero matrix where one
    element is one. ``Ĝ(f) = C*(z(f)*I - A)^(-1)*B + D`` is the estimated and
    ``G(f)`` is the measured frequency response matrix (FRM).

    The structure of the Jacobian is: ``JX[f,p,m,i]`` where ``p`` and ``m`` are
    inputs and outputs and ``f`` the frequency line. ``i`` is the index
    mapping, relating the matrix element ``(k,l)`` of ``X`` to the linear index
    of the vector ``JX[p,m,:,f]``. This mapping is given by, fx for ``A``:
    ``i = np.ravel_multi_index((k,l) ,(n,n))`` and the reverse is
    ``k, l = np.unravel_index(i, (n,n))``. Thus ``JA(f,:,:,i)`` contains the
    partial derivative of the unweighted error ``e(f)`` at frequency `f` wrt.
    ``A(k,l)``

    Parameters
    ----------
    A : ndarray(n,n)
        state matrix
    B : ndarray(n,m)
        input matrix
    C : ndarray(p,n)
        output matrix
    z : ndarray(F)
        ``z = exp(2j*pi*freq)``, where freq is a vector of normalized
        frequencies at which the Jacobians are computed (0 < freq < 0.5)

    Returns
    -------
    JA : ndarray(F,p,m,n*n)
        JA(f,:,:,i) contains the partial derivative of the unweighted error
        e(f) at frequency f wrt. A(k,l)
    JB : ndarray(p,m,n*m,F)
        JB(f,:,:,i) contains the partial derivative of e(f) w.r.t. B(k,l)
    JC : ndarray(p,m,p*n,F)
        JC(f,:,:,i) contains the partial derivative of e(f) w.r.t. C(k,l)

    Notes
    -----
    See eq. (5-103) in :cite:pauldart2008

    """

    F = len(z)          # Number of frequencies
    n = np.shape(A)[0]  # Number of states
    m = np.shape(B)[1]  # Number of inputs
    p = np.shape(C)[0]  # Number of outputs

    JA = np.empty((F,p,m,n*n),dtype=complex)
    JB = np.empty((F,p,m,n*m),dtype=complex)
    JC = np.empty((F,p,m,n*p),dtype=complex)

    # get rows and columns in A for a given index: A(i)=A(k(i),ell(i))
    k, ell = np.unravel_index(np.arange(n**2), (n,n))
    # Note that using inv(A) implicitly calls solve and creates an identity
    # matrix. Thus it is faster to allocate In once and then call solve.
    In = np.eye(n)
    Im = np.eye(m)
    Ip = np.eye(p)
    # TODO must vectorize...
    # see for calling lapack routines directly
    # https://stackoverflow.com/a/11999063/1121523
    # see for multicasting
    # https://docs.scipy.org/doc/numpy/reference/routines.linalg.html#linear-algebra-on-several-matrices-at-once
    for f in range(F):
        temp1 = solve((z[f]*In - A),In)
        temp2 = C @ temp1
        temp3 = temp1 @ B

        # Jacobian w.r.t. all elements in A, A(i)=A(k(i),ell(i))
        # Note that the partial derivative of e(f) w.r.t. A(k(i),ell(i)) is
        # equal to temp2*fOne(n,n,i)*temp3, and thus
        # JA(:,:,i,f) = temp2(:,k(i))*temp3(ell(i),:)
        for i in range(n**2): # Loop over all elements in A
            JA[f,:,:,i] = np.outer(temp2[:,k[i]], temp3[ell[i],:])

        # Jacobian w.r.t. all elements in B
        # Note that the partial derivative of e(f) w.r.t. B(k,l) is equal to
        # temp2*fOne(n,m,sub2ind([n m],k,l)), and thus
        # JB(:,l,sub2ind([n m],k,l),f) = temp2(:,k)
        JB[f] = np.reshape(kron(Im, temp2), (p,m,m*n))

        # Jacobian w.r.t. all elements in C
        # Note that the partial derivative of e(f) w.r.t. C(k,l) is equal to
        # fOne(p,n,sub2ind([p n],k,l))*temp3, and thus
        # JC(k,:,sub2ind([p n],k,l),f) = temp3(l,:)
        JC[f] = np.reshape(kron(temp3.T, Ip), (p,m,n*p))

    # JD does not change over iterations
    JD = np.zeros((p,m,p*m))
    for f in range(p*m):
        np.put(JD[...,f], f, 1)
    JD = np.tile(JD, (F,1,1,1))

    return JA, JB, JC, JD

def modal_list(G, covG, freq, nvec, r, fs, U=None, Y=None):
    """Calculate modal properties for list of system size ``n``

    Used for creating stabilization diagram.

    Returns
    -------
    dict of dicts with modal properties
    """

    if U is None and Y is None:
        F,p,m = G.shape
    else:
        p = Y.shape[1]

    nmax = np.max(nvec)
    sqrtCY, U, s = subspace(G, covG, freq, nmax, r, U, Y, modal=True)

    # estimate modal properties for increasing model order
    md = {}
    for n in sorted(nvec):
        # Estimation of the extended observability matrix Or, eq (21)
        Or = sqrtCY @ U[:,:n]  # @ np.diag(np.sqrt(s[:n]))

        # Estimate A from eq(24) and C as the first block row of Or.
        A, *_ = lstsq(Or[:-p,:], Or[p:,:])
        C = Or[:p, :]
        stable = is_stable(A, domain='z')
        # Convert A into continous-time arrays using eq (8)
        A = fs * logm(A)
        modal = modal_ac(A, C)
        md[n] = {**modal, 'stable': stable}

    return md

def subspace(G, covG, freq, n, r, U=None, Y=None, bd_method='nr',
             modal=False):
    """Estimate state-space model from Frequency Response Function (or Matrix)

    The linear state-space model is estimated from samples of the frequency
    response function (or frequency response matrix). The frequency-domain
    subspace method in `McKelvey1996`_ is applied with the frequency weighting
    in `Pintelon2002`_, i.e. weighting with the sampled covariance matrix.

    `p`: number of outputs, `m`: number of inputs, `F`: number of frequencies.

    Parameters
    ----------
    G : complex ndarray(p, m, F)
        Frequency Response Matrix (FRM)
    covG : ndarray(p*m, p*m, F)
        σ²_G, Covariance tensor on G (False if no weighting required)
    freq : ndarray(F)
        Vector of normalized frequencies at which the FRM is given (0 < freq < 0.5)
    n : int
        Model order
    r : int
        Number of block rows in the extended observability matrix (r > n)
    bd_method : str {'nr', 'explicit'}, optional
        Method used for BD estimation
    modal : bool {false}, optional
        Return

    Returns
    -------
    A : ndarray(n, n)
        state matrix
    B : ndarray(n, m)
        input matrix
    C : ndarray(p, n)
        output matrix
    D : ndarray(p, m)
        feed-through matrix
    unstable : boolean
        Indicating whether or not the identified state-space model is unstable

    Notes
    -----
    Algorithm: (see p. 119 `Paduart2008`_ for details)
    From a DFT of the state space eqs., and recursive use of the two equations
    give the relation: ``Gmat = OᵣX + SᵣU``. From this ``A`` and ``C`` are
    determined. ``B`` and ``D`` are found by minimizing the weighted error
    ``e(f) = W*(Ĝ(f) - G(f))`` where ``Ĝ(f) = C*(z(f)*I - A)^(-1)*B + D`` is
    the estimated- and ``G(f)`` is the measured frequency response matrix(FRM).
    The weight, ``W=1/σ_G``, is chosen in :cite:pinleton2002, sec. 5, to almost
    eliminate the bias resulting from observing the inputs and outputs ``U``
    and ``Y`` with errors.

    In ``Gmat``, ``Sᵣ`` is a lower triangular block toeplitz matrix and ``Oᵣ``,
    ``U`` are extended matrices and found as:
      1. Construct Extended observability matrix Oᵣ
          a. Construct Wᵣ with z
          b. Construct Hmat with H and Wᵣ
          c. Construct Umat with Wᵣ (U=eye(m))
          d. Split real and imaginary parts of Umat and Hmat
          e. Z=[Umat; Hmat]
          f. Calculate CY
          g. QR decomposition of Zᵀ (orthogonal projection)
          h. CY^(-1/2)*RT22=USV'
          i. Oᵣ=U(:,1:n)
      2. Estimate A and C from the shift property of Oᵣ
      3. Estimate B and D given A,C and H

    References
    ----------
    .. _McKelvey1996:
       McKelvey T., Akcay, H., and Ljung, L. (1996).
       Subspace-Based Multivariable System Identification From Frequency
       Response Data. IEEE Transactions on Automatic Control, 41(7):960-979

    .. _Pintelon2002:
       Pintelon, R. (2002). Frequency-domain subspace system identification
       using non-parametric noise models. Automatica, 38:1295-1311

    .. _Paduart2008:
       Paduart J. (2008). Identification of nonlinear systems using polynomial
       nonlinear state space models. PhD thesis, Vrije Universiteit Brussel.

    .. _noel2013:
       Noël, J.P., Kerschen G. (2013)
       Frequency-domain subspace identification for nonlinear mechanical
       systems. MSSP, doi:10.1016/j.ymssp.2013.06.034

    """
    # number of outputs/inputs and number of frequencies
    # When using G as input, _m reflects that G is 3d: (F,p,m), ie U: (F,m)
    if U is None and Y is None:
        F,p,m = G.shape
        is_frf = True
        _m = m
    else:
        F = len(freq)
        p = Y.shape[1]
        m = U.shape[1]
        is_frf = False
        _m = 1

    # 1.a. Construct Wr with z
    z = np.exp(2j*np.pi*freq)
    # if B,D is calculated explicit, we need an additional p and m rows in Gmat
    # and Umat. See eq (30) in noel2013.
    expl = 0
    if bd_method == 'explicit':
        expl = 1

    Wr = (z[:,None]**np.arange(r+expl)).T
    # 1.b. and 1.c. Construct Gmat and Umat
    # The shape depends on the method, ie if Y,U or G is supplied
    Gmat = np.empty(((r+expl)*p,F*_m), dtype=complex)
    Umat = np.empty(((r+expl)*m,F*_m), dtype=complex)
    if U is None and Y is None:
        for f in range(F):
            Gmat[:,f*m:(f+1)*m] = kron(Wr[:,f,None], G[f])
            Umat[:,f*m:(f+1)*m] = kron(Wr[:,f,None], np.eye(m))
    else:
        for f in range(F):
            Gmat[:,f] = kron(Wr[:,f], Y[f])
            Umat[:,f] = kron(Wr[:,f], U[f])

    # 1.e. and 1.f: split into real and imag part and stack into Z
    # we do it in a memory efficient way and avoids intermediate memory copies.
    # (Just so you know: It is more efficient to stack the result in a new
    # memory location, than overwriting the old). Ie.
    # Gre = np.hstack([Gmat.real, Gmat.imag]) is more efficient than
    # Gmat = np.hstack([Gmat.real, Gmat.imag])
    Z = np.empty(((r+expl)*(p+m), 2*F*_m))
    Z[:(r+expl)*m,:F*_m] = Umat.real
    Z[:(r+expl)*m,F*_m:] = Umat.imag
    Z[(r+expl)*m:,:F*_m] = Gmat.real
    Z[(r+expl)*m:,F*_m:] = Gmat.imag

    # 1.f. Calculate CY from σ²_G
    if covG is False or covG is None:
        CY = np.eye(p*r)
        # covG = np.tile(np.eye(p*m), (F,1,1))
    else:
        CY = np.zeros((p*r,p*r))
        for f in range(F):
            # Take sum over the diagonal blocks of cov(vec(H)) (see
            # paduart2008(5-93))
            temp = np.zeros((p,p),dtype=complex)
            for i in range(m):
                temp += covG[f, i*p:(i+1)*p, i*p:(i+1)*p]
                CY += np.real(kron(np.outer(Wr[:r,f], Wr[:r,f].conj()),temp))

    # 1.g. QR decomposition of Z.T, Z=R.T*Q.T, to eliminate U from Z.
    R = qr(Z.T, mode='r')
    RT = R.T
    if bd_method == 'explicit':
        RT22 = RT[-(r+1)*p:-p,-(r+1)*p:-p]
    else:
        RT22 = RT[-r*p:,-r*p:]

    # 1.h. CY^(-1/2)*RT22=USV', Calculate CY^(-1/2) using svd decomp.
    UC, sc, _ = svd(CY, full_matrices=False)

    # it is faster to work on the diagonal scy, than the full matrix SCY
    # Note: We work with real matrices here, thus UC.conj().T -> UC.T
    sqrtCY = UC * np.sqrt(sc) @ UC.conj().T
    invsqrtCY = UC * 1/np.sqrt(sc) @ UC.conj().T

    # Remove noise. By taking svd of CY^(-1/2)*RT22
    Un, sn, _ = svd(invsqrtCY @ RT22)  # , full_matrices=False)

    if modal:
        # in case we want to calculate A, C for different n's
        return sqrtCY, Un, sn

    if n == 0:
        # Offer possibility to choose model order
        n = int(input('Input model size'))

    # 1.i. Estimate extended observability matrix
    Or = sqrtCY @ Un[:,:n]  # @ np.diag(np.sqrt(sn[:n]))

    # 2. Estimate A and C from shift property of Or
    A, *_ = lstsq(Or[:-p,:], Or[p:,:])
    C = Or[:p,:].copy()
    # Recompute Or from A and C. Or plays a major role in determining B
    # and D, thus J.P. Noel suggest that Or might be recalculated
    # Equal to Or[] = C @ np.linalg.matrix_power(A,j)
    # for j in range(1,r):
    #     Or[j*p:(j+1)*p,:] = Or[(j-1)*p:j*p,:] @ A

    # 3. Estimate B and D given A,C and H: (W)LS estimate
    # Compute weight, W = sqrt(σ²_G^-1)
    weight = False
    if covG is not False and covG is not None:
        weight = np.zeros_like(covG)  # .transpose((2,0,1))
        for f in range(F):
            weight[f] = matrix_square_inv(covG[f])

    if bd_method == 'explicit':
        B, D = bd_explicit(A,C,Or,n,r,m,p,RT)
    else:  # bd_method == 'nr':
        B, D = bd_nr(A,C,G,freq,n,r,m,p,U,Y,weight)

    # Check stability of the estimated model
    isstable = is_stable(A)
    return A, B, C, D, z, isstable

def bd_explicit(A,C,Or,n,r,m,p,RT):
    """Estimate B, D using explicit solution

    """
    # R_U: Ui+1, R_Y: Yi+1
    R_U = RT[:m*(r+1),:(m+p)*(r+1)]
    R_Y = RT[m*(r+1):(m+p)*(r+1),:(m+p)*(r+1)]

    # eq. 30
    Or_inv = pinv(Or)
    Q = np.vstack([
        Or_inv @ np.hstack([np.zeros((p*r,p)), np.eye(p*r)]) @ R_Y,
        R_Y[:p,:]]) - \
        np.vstack([
            A,
            C]) @ Or_inv @ np.hstack([np.eye(p*r), np.zeros((p*r,p))]) @ R_Y

    Rk = R_U

    # eq (34) with zeros matrix appended to the end. eq. L1,2 = [L1,2, zeros]
    L1 = np.hstack([A @ Or_inv, np.zeros((n,p))])
    L2 = np.hstack([C @ Or_inv, np.zeros((p,p))])

    # The pseudo-inverse of G. eq (33), prepended with zero matrix.
    M = np.hstack([np.zeros((n,p)), Or_inv])

    # The reason for appending/prepending zeros in P and M, is to easily
    # form the submatrices of N, given by eq. 40. Thus ML is equal to first
    # row of N1
    ML = M - L1

    # rhs multiplicator of eq (40)
    Z = np.vstack([
        np.hstack([np.eye(p), np.zeros((p,n))]),
        np.hstack([np.zeros((p*r,p)), Or])
    ])

    # Assemble the kron_prod in eq. 44.
    for j in range(r+1):
        # Submatrices of N_k. Given by eq (40).
        # eg. N1 corresspond to first row, N2 to second row of the N_k's
        # submatrices
        N1 = np.zeros((n,p*(r+1)))
        N2 = np.zeros((p,p*(r+1)))

        N1[:, :p*(r-j+1)] = ML[:, j*p:p*(r+1)]
        N2[:, :p*(r-j)] = -L2[:, j*p:p*r]

        if j == 0:
            N2[:p, :p] += np.eye(p)

        # Evaluation of eq (40)
        Nk = np.vstack([N1, N2]) @ Z

        if j == 0:
            kron_prod = np.kron(Rk[j*m:(j+1)*m,:].T, Nk)
        else:
            kron_prod += np.kron(Rk[j*m:(j+1)*m,:].T, Nk)

    DB, *_ = lstsq(kron_prod, Q.ravel(order='F'), check_finite=False)
    DB = DB.reshape(n+p,m, order='F')
    D = DB[:p,:]
    B = DB[p:,:]

    return B, D

def bd_nr(A,C,G,freq,n,r,m,p,U=None,Y=None,weight=False):
    """Estimate B, D using transfer function-based optimization
    (Newton-Raphson iterations)

    """

    # initial guess for B and D
    theta = np.zeros((m*(n+p)))
    niter = 1  # one iteration is enough
    for i in range(niter):
        if U is None and Y is None:
            cost = frf_costfcn(theta, G, weight)
        else:
            cost = output_costfcn(theta,A,C,n,m,p,freq,U,Y,weight)
        jac = frf_jacobian(theta,A,C,n,m,p,freq,U,weight)

        # Normalize columns of Jacobian with their rms value. Might increase
        # numerical condition
        jac, scaling = normalize_columns(jac)
        # Compute Gauss-Newton parameter update. For small residuals e(Θ), the
        # Hessian can be approximated by the Jacobian of e. See (5.140) in
        # paduart2008.
        dtheta, res, rank, s = lstsq(jac, cost, check_finite=False)
        dtheta /= scaling
        theta += dtheta

    B = theta[:n*m].reshape((n,m))
    D = theta[n*m:].reshape((p,m))

    return B, D

def frf_costfcn(x0, G, weight=False):
    """Compute the cost, V = -W*(0 - G), i.e. minus the weighted error(the
    residual) when considering zero initial estimates for Ĝ

    Using x0, it would be possible to extract B, D and create an estimate G
    Ĝ(f) = C*inv(z(f)*I - A)*B + D and W = 1/σ_G.

    """
    if weight is False:
        resG = G.ravel()
    else:
        resG = mmul_weight(G, weight).ravel()

    return np.hstack((resG.real, resG.imag))

def frf_jacobian(x0,A,C,n,m,p,freq,U=None,weight=False):
    """Compute partial derivative of the weighted error, e = W*(Ĝ - G) wrt B, D

    Ĝ(f) = C*inv(z(f)*I - A)*B + D and W = 1/σ_G.

    For FNSI:
    Compute partial derivative of the weighted error, e = W*(Ŷ - Y) wrt B, D

    Ĝ(f) = C*inv(z(f)*I - A)*B + D and W = 1/σ_G.
    Ŷ(f) = U(f) * Ĝ(f)

    """

    z = np.exp(2j*np.pi*freq)
    B = x0[:n*m].reshape(n,m)
    D = x0[n*m:m*(n+p)].reshape(p,m)
    # unweighted jacobian
    _, JB, _, JD = jacobian_freq(A,B,C,z)
    # add weight
    F = len(z)
    npar = m*(n+p)
    if U is None:
        _m = m
        tmp = np.empty((F,p,m,npar),dtype=complex)
        tmp[...,:n*m] = JB
        tmp[...,n*m:] = JD
    else:
        _m = 1
        tmp = np.empty((F,p,npar),dtype=complex)
        # fast way of doing: JB[f] = U[f] @ JB[f].T
        tmp[...,:n*m] = np.einsum('ij,iljk->ilk',U,JB)
        tmp[...,n*m:] = np.einsum('ij,iljk->ilk',U,JD)

    tmp.shape = (F,p*_m,npar)
    if weight is not False:
        tmp = mmul_weight(tmp, weight)
    tmp.shape = (F*p*_m,npar)

    # B and D as one parameter vector => concatenate Jacobians
    # We do: J = np.hstack([JB, JD]), jac = np.vstack([J.real, J.imag]), but
    # faster
    jac = np.empty((2*F*p*_m, npar))
    jac[:F*p*_m] = tmp.real
    jac[F*p*_m:] = tmp.imag

    return jac

def output_costfcn(x0,A,C,n,m,p,freq,U,Y,weight):
    """Compute the cost, e = W*(Ŷ - Y)

    Ĝ(f) = C*inv(z(f)*I - A)*B + D and W = 1/σ_G.
    Ŷ(f) = U(f) * Ĝ(f)
    """
    B = x0[:n*m].reshape(n,m)
    D = x0[n*m:m*(n+p)].reshape(p,m)

    Gss = ss2frf(A,B,C,D,freq)
    Gss = np.random.rand(*Gss.shape)
    # fast way of doing: Ymodel[f] = U[f] @ Gss[f].T
    Ymodel = np.einsum('ij,ilj->il',U,Gss)
    V = Ymodel - Y
    if weight is not False:
        V = V.ravel(order='F')
    else:
        # TODO order='F' ?
        V = mmul_weight(V, weight).ravel()

    return np.hstack((V.real, V.imag))

def jacobian(x0, system, weight=False):

    n, m, p, npar = system.n, system.m, system.p, system.npar
    F = len(system.z)

    A, B, C, D = system.extract(x0)
    JA, JB, JC, JD = jacobian_freq(A,B,C,system.z)

    tmp = np.empty((F,p,m,npar),dtype=complex)
    tmp[...,:n**2] = JA
    tmp[...,n**2 +       np.r_[:n*m]] = JB
    tmp[...,n**2 + n*m + np.r_[:n*p]] = JC
    tmp[...,n**2 + n*m + n*p:] = JD
    tmp.shape = (F,m*p,npar)

    if weight is not False:
        tmp = mmul_weight(tmp, weight)
    tmp.shape = (F*p*m,npar)

    jac = np.empty((2*F*p*m, npar))
    jac[:F*p*m] = tmp.real
    jac[F*p*m:] = tmp.imag

    return jac

def costfcn(x0, system, weight=False):
    """Compute the error vector of the FRF, such that the function to mimimize is

    res = ∑ₖ e[k]ᴴ*e[k], where the error is given by
    e = weight*(Ĝ - G)
    and the weight is the square inverse of the covariance matrix of `G`,
    weight = \sqrt(σ_G⁻¹) Ĝ⁻¹

    """
    A, B, C, D = system.extract(x0)

    # frf of the state space model
    Gss = ss2frf(A,B,C,D,system.signal.norm_freq)
    err = Gss - system.signal.G
    if weight is not False:
        err = mmul_weight(err, weight)
    err_w = np.hstack((err.real.ravel(), err.imag.ravel()))

    return err_w  # err.ravel()

def extract_model(models, y, u, dt, t=None, x0=None):
    """extract the best model using validation data"""

    dictget = lambda d, *k: [d[i] for i in k]
    err_old = np.inf
    err_vec = np.empty(len(models))
    for i, (k, model) in enumerate(models.items()):

        A, B, C, D = dictget(model, 'A', 'B', 'C', 'D')
        system = (A, B, C, D, dt)
        tout, yout, xout = dlsim(system, u, t, x0)
        err_rms = np.sqrt(np.mean((y - yout)**2))
        err_vec[i] = err_rms
        if err_rms < err_old:
            n = k
            err_old = err_rms

    return models[n], err_vec
