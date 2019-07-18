#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import itertools
from scipy.linalg import svd, norm


# general messages for LM/etc optimization
TERMINATION_MESSAGES = {
    None: "Status returned `None`. Error.",
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of iterations is exceeded.",
    1: "`gtol` termination condition is satisfied. (small change in Jacobian)",
    2: "`ftol` termination condition is satisfied. (small change in cost)",
    3: "`xtol` termination condition is satisfied. (small step)",
    4: "Both `ftol`(cost) and `xtol`(step) termination conditions are satisfied."
}

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def next_pow2(i):
    """
    Find the next power of two

    >>> int(next_pow2(5))
    8
    >>> int(next_pow2(250))
    256
    """
    # do not use NumPy here, math is much faster for single values
    exponent = math.ceil(math.log(i) / math.log(2))
    # the value: int(math.pow(2, exponent))
    return exponent

def prime_factor(n):
    """Find the prime factorization of n

    Efficient implementation. Find the factorization by trial division, using
    the optimization of dividing only by two and the odd integers.

    An improvement on trial division by two and the odd numbers is wheel
    factorization, which uses a cyclic set of gaps between potential primes to
    greatly reduce the number of trial divisions. Here we use a 2,3,5-wheel

    Factoring wheels have the same O(sqrt(n)) time complexity as normal trial
    division, but will be two or three times faster in practice.

    >>> list(factors(90))
    [2, 3, 3, 5]
    """
    f = 2
    increments = itertools.chain([1,2,2], itertools.cycle([4,2,4,2,4,6,2,6]))
    for incr in increments:
        if f*f > n:
            break
        while n % f == 0:
            yield f
            n //= f
        f += incr
    if n > 1:
        yield n

def db(x, r=1):
    """relative value in dB

    TODO: Maybe x should be rescaled to ]0..1].?
    log10(0) = inf.

    Parameters
    ----------
    x: array like
    r: float, optional
        Reference value. default = 1

    Notes
    -----
    https://en.wikipedia.org/wiki/Decibel#Field_quantities_and_root-power_quantities
    """
    if not math.isclose(r, 1, rel_tol=1e-6):
        x = x/r

    # dont nag if x=0
    with np.errstate(divide='ignore', invalid='ignore'):
        return 20*np.log10(np.abs(x))


def import_npz(npz_file, namespace=globals()):
    """Load npz file and unpack data/dictionary to the given namespace

    It is necessary to explicit call the function with globals() even if it is
    set as default value here. The docs states that the scope is the defining
    module not the calling.

    Example for `oneliner` without using namespace(can only be used local)
    for varName in data.files:
        exec(varName + " = data['" + varName + "']")

    Notes:
    ------
    https://docs.python.org/3/library/functions.html#globals
    """
    data = np.load(npz_file)
    for varName in data:
        try:
            namespace[varName] = data[varName].item()
        except ValueError:
            namespace[varName] = data[varName]

def window(iterable, n=3):
    """Returns a sliding window (of width n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ..."""
    # https://stackoverflow.com/a/6822773/1121523
    it = iter(iterable)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for element in it:
        result = result[1:] + (element,)
        yield result

def rescale(x, mini=None, maxi=None):
    """Rescale x to 0-1.

    If mini and maxi is given, then they are used as the values that get scaled
    to 0 and 1, respectively

    Notes
    -----
    To 0..1:
    z_i = (x_i− min(x)) / (max(x)−min(x))

    Or custom range:
    a = (maxval-minval) / (max(x)-min(x))
    b = maxval - a * max(x)
    z = a * x + b

    """
    if hasattr(x, "__len__") is False:
        return x

    if mini is None:
        mini = np.min(x)
    if maxi is None:
        maxi = np.max(x)
    return (x - mini) / (maxi - mini)

def meanVar(Y, isnoise=False):
    """
    Y = fft(y)/nsper

    Parameters
    ----------
    Y : ndarray (ndof, nsper, nper)
        Y is the fft of y
    """

    # number of periods
    p = Y.shape[2]

    # average over periods
    Ymean = np.sum(Y,axis=2) / p

    # subtract column mean from y in a broadcast way. Ie: y is 3D matrix and
    # for every 2D slice we subtract y_mean. Python automatically
    # broadcast(repeat) y_mean.
    # https://scipy.github.io/old-wiki/pages/EricsBroadcastingDoc
    Y0 = Y - Ymean[...,None]

    W = []
    # weights. Only used if the signal is noisy and multiple periods are
    # used
    if p > 1 and isnoise:
        W = np.sum(np.abs(Y0)**2, axis=2)/(p-1)

    return Ymean, W


def weightfcn(cov):
    """Calculate weight. For subspace is the square inverse of covG. For
    pnlss it is the square inverse of covY"""
    F = cov.shape[0]
    covinvsq = np.empty_like(cov)
    for f in range(F):
        covinvsq[f] = matrix_square_inv(cov[f])
    return covinvsq

def matrix_square_inv(A):
    """Calculate the inverse of the matrix square root of `A`
    Calculate `X` such that XX = inv(A)
    `A` is assumed positive definite, thus the all singular values are strictly
    positive. Given the svd decomposition A=UsVᴴ, we see that
    AAᴴ = Us²Uᴴ (remember (UsV)ᴴ = VᴴsUᴴ) and it follows that
    (AAᴴ)⁻¹/² = Us⁻¹Uᴴ
    Returns
    -------
    X : ndarray(n,n)
       Inverse of matrix square root of A
    Notes
    -----
    See the comments here.
    https://math.stackexchange.com/questions/106774/matrix-square-root
    """
    U, s, _ = svd(A, full_matrices=False)
    return U * 1/np.sqrt(s) @ U.conj().T

def mmul_weight(mat, weight):
    """Add weight. Computes the Jacobian of the weighted error ``e_W(f) = W(f,:,:)*e(f)``

    """
    # np.einsum('ijk,kl',weight, mat) or
    # np.einsum('ijk,kl->ijl',weight, mat) or
    # np.einsum('ijk,jl->ilk',weight,mat)
    # np.tensordot(weight, mat, axes=1)
    # np.matmul(weight, mat)
    return np.matmul(weight, mat)

def normalize_columns(mat):

    # Rms values of each column
    scaling = np.sqrt(np.mean(mat**2,axis=0))
    # or scaling = 1/np.sqrt(mat.shape[0]) * np.linalg.norm(mat,ord=2,axis=0)
    # Robustify against columns with zero rms value
    scaling[scaling == 0] = 1
    # Scale columns with 1/rms value
    # This modifies mat in place(ie the input mat). We do not want that.
    # mat /= scaling
    return mat/scaling, scaling

def lm(fun, x0, jac, info=2, nmax=50, lamb=None, ftol=1e-8, xtol=1e-8,
       gtol=1e-8, args=(), kwargs={}):
    """Solve a nonlinear least-squares problem using levenberg marquardt
       algorithm. See also :scipy-optimize:func:`scipy.optimize.least_squares`

    Parameters
    ----------
    fun : callable
        Function which computes the vector of residuals
    x0: array_like with shape (n,) or float
        Initial guess on independent variables.
    jac : callable
        Method of computing the Jacobian matrix (an m-by-n matrix, where
        element (i, j) is the partial derivative of f[i] with respect to
        x[j]).
    ftol : float, optional
        Tolerance for termination by the change of the cost function. Default
        is 1e-8. The optimization process is stopped when  ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step.
    xtol : float, optional
        Tolerance for termination by the change of the independent variables.
        Default is 1e-8.
    gtol : float, optional
        Tolerance for termination by the norm of the gradient. Default is 1e-8.
    info : {0, 1, 2}, optional
        Level of algorithm's verbosity:
            * 0 (default) : work silently.
            * 1 : display a termination report.
            * 2 : display progress during iterations

    """
    # the error vector
    err_old = fun(x0, *args, **kwargs)
    # Maybe divide by 2 to match scipy's implementation of minpack
    cost = np.dot(err_old, err_old)
    cost_old = cost.copy()

    # Initialization of the Levenberg-Marquardt loop
    niter = 0
    ninner_max = 10
    nfev = 1
    status = None
    message = ''
    cost_vec = np.empty(nmax+1)
    x0_mat = np.empty((nmax+1, len(x0)))
    # save initial guess
    x0_mat[0] = x0.copy()
    cost_vec[0] = cost.copy()

    if info == 2:
        print(f"{'i':3} | {'inner':5} | {'cost':12} | {'cond':12} |"
              f" {'lambda':6}")

    stop = False
    while niter < nmax and not stop:

        J = jac(x0, *args, **kwargs)
        J, scaling = normalize_columns(J)
        U, s, Vt = svd(J, full_matrices=False)

        if norm(J) < gtol:  # small jacobian
            stop = True
            status = 1

        if lamb is None:
            # Initialize lambda as largest sing. value of initial jacobian.
            # pinleton2002
            lamb = s[0]

        # as long as the step is unsuccessful
        ninner = 0
        # determine rank of jacobian/estimate non-zero singular values(rank
        # estimate)
        tol = max(J.shape)*np.spacing(max(s))
        r = np.sum(s > tol)

        # step with direction from err
        s = s[:r]
        sr = s.copy()  # only saved to calculate cond. number later
        while cost >= cost_old and ninner < ninner_max and not stop:
            s /= (s**2 + lamb**2)
            ds = -np.linalg.multi_dot((err_old, U[:,:r] * s, Vt[:r]))
            ds /= scaling

            x0test = x0 + ds
            err = fun(x0test, *args, **kwargs)
            cost = np.dot(err,err)

            if cost >= cost_old:
                # step unsuccessful, increase lambda, ie. Lean more towards
                # gradient descent method(converges in larger range)
                lamb *= np.sqrt(10)
                s = sr.copy()
            elif np.isnan(cost):
                print('Unstable model. Increasing lambda')
                cost = np.inf
                lamb *= np.sqrt(10)
                s = sr.copy()
            else:
                # Lean more towards Gauss-Newton algorithm(converges faster)
                lamb /= 2
            ninner += 1

            if norm(ds) < xtol:  # small step
                stop = True
                status = 3
            if np.abs((cost-cost_old)/cost) < ftol:  # small change in costfcn
                stop = True
                status = 2 if status is None else 4

        if info == 2:
            jac_cond = sr[0]/sr[-1]
            # {cost/2/nfd/R/p:12.3f} for freq weighting
            print(f"{niter:3d} | {ninner:5d} | {cost:12.8g} | {jac_cond:12.3f}"
                  f" | {lamb:6.3f}")

        if cost < cost_old or stop:
            cost_old = cost
            err_old = err
            x0 = x0test
            # save intermediate models
            x0_mat[niter+1] = x0.copy()
            cost_vec[niter+1] = cost.copy()

        niter += 1
        nfev += ninner

    if niter == nmax:
        status = 0
    message = TERMINATION_MESSAGES[status]
    if info > 0:
        print(f"Terminated: {message:s}")
        print(f"Function evaluations {nfev}, initial cost {cost_vec[0]:.4e}, "
              f"final cost {cost:.4e}")

    res = {'x':x0, 'cost': cost, 'fun':err, 'niter': niter, 'x_mat':
           x0_mat[:niter], 'cost_vec':cost_vec[niter], 'message':message,
           'success':status > 0, 'nfev':nfev, 'njev':niter, 'status':status}
    return res
