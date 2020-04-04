import numpy as np
from scipy import integrate
MAXITER = 12


def c_nk(n: int, k: int):
    """
    Compute binomial coefficient 
    """
    assert n >= k and k >= 0, f"n must be >= k and k must be >= 0"
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))


def moments(max_s: int, xl: float, xr: float, a: float = 0.0, b: float = 1.0, alpha: float = 0.0, beta: float = 0.0):
    """
    compute moments of the weight 1 / (x-a)^alpha / (b-x)^beta over [xl, xr]
    max_s : highest required order
    xl : left limit
    xr : right limit
    a : weight parameter a
    b : weight parameter b
    alpha : weight parameter alpha
    beta : weight parameter beta
    """

    assert alpha * beta == 0, \
        f'alpha ({alpha}) and/or beta ({beta}) must be 0'

    if alpha == 0 and beta == 0:
        return [(xr ** s - xl ** s) / s for s in range(1, max_s + 2)]

    mu = np.zeros(max_s + 1)

    if alpha == 0:
        for j in range(max_s + 1):
            mu[j] = -sum([(-1) ** (k+1) * c_nk(j, k) * b ** (j - k) * (a - b)
                          ** (k - beta + 1) / (k - beta + 1) for k in range(j + 1)])

    if beta == 0:
        for j in range(max_s + 1):
            mu[j] = sum([c_nk(j, k) * a ** (j - k) * (b - a) **
                         (k - alpha + 1) / (k - alpha + 1) for k in range(j + 1)])

    return mu


def quad(f, xl: float, xr: float, nodes, *params):
    """
    small Newton—Cotes formula
    f: function to integrate
    xl : left limit
    xr : right limit
    nodes: nodes within [xl, xr]
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mu = moments(len(nodes) - 1, xl, xr, *params)

    X = np.array([[nodes[j]**s for j in range(len(nodes))]
                  for s in range(len(nodes))])
    A = np.linalg.solve(X, mu)

    return sum([A[j] * f(nodes[j]) for j in range(len(nodes))])


def quad_gauss(f, xl: float, xr: float, n: int, *params):
    """
    small Gauss formula
    f: function to integrate
    xl : left limit
    xr : right limit
    n : number of nodes
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mu = np.array(moments(2 * n - 1, xl, xr, *params))

    raise NotImplementedError
    # return # small formula result over [xl, xr]


def composite_quad(f, xl: float, xr: float, N: int, n: int, *params):
    """
    composite Newton—Cotes formula
    f: function to integrate
    xl : left limit
    xr : right limit
    N : number of steps
    n : number of nodes od small formulae
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mesh = np.linspace(xl, xr, N + 1)
    return sum(quad(f, mesh[i], mesh[i + 1], equidist(n, mesh[i], mesh[i + 1]), *params) for i in range(N))


def composite_gauss(f, a: float, b: float, N: int, n: int, *params):
    """
    composite Gauss formula
    f: function to integrate
    xl : left limit
    xr : right limit
    N : number of steps
    n : number of nodes od small formulae
    *params: parameters of the variant — a, b, alpha, beta)
    """
    mesh = np.linspace(a, b, N + 1)
    return sum(quad_gauss(f, mesh[i], mesh[i + 1], n, *params) for i in range(N))


def equidist(n: int, xl: float, xr: float):
    if n == 1:
        return [0.5 * (xl + xr)]
    else:
        return np.linspace(xl, xr, n)


def runge(s1: float, s2: float, L: float, m: float):
    """ estimate m-degree error for s2 """
    raise NotImplementedError


def aitken(s1: float, s2: float, s3: float, L: float):
    """
    estimate convergence order
    s1, s2, s3: consecutive composite quads
    return: convergence order estimation
    """
    # if NON MONOTONOUS CONVERGENCE
    #    return -1
    raise NotImplementedError


def doubling_nc(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with theoretical convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter : number of iterations (steps doubling)
    iter = 0

    if iter == MAXITER:
        print("Convergence not reached!")
        return 0, 0, 10*tol

    raise NotImplementedError
    return N, S, err


def doubling_nc_aitken(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with Aitken estimation of the convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # m : estimated convergence rate by Aitken for S
    # iter : number of iterations (steps doubling)
    iter = 0

    if iter == MAXITER:
        print("Convergence not reached!")
        return 0, 0, 10*tol, -100

    raise NotImplementedError
    return N, S, err, m


def doubling_gauss(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with theoretical convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter : number of iterations (steps doubling)
    iter = 0

    if iter == MAXITER:
        print("Convergence not reached!")
        return 0, 0, 10*tol

    raise NotImplementedError
    return N, S, err


def doubling_gauss_aitken(f, xl: float, xr: float, n: int, tol: float, *params):
    """
    compute integral by doubling the steps number with Aitken estimation of the convergence rate
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # m : estimated convergence rate by Aitken for S
    # iter : number of iterations (steps doubling)
    iter = 0

    if iter == MAXITER:
        print("Convergence not reached!")
        return 0, 0, 10*tol, -100

    raise NotImplementedError
    return N, S, err, m


def optimal_nc(f, xl: float, xr: float, n: int, tol: float, *params):
    """ estimate the optimal step with Aitken and Runge procedures
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """
    # required local variables to return
    # S : computed value of the integral with required tolerance
    # N : number of steps for S
    # err : estimated error of S
    # iter : number of iterations (steps doubling)
    iter = 0

    if iter == MAXITER:
        print("Convergence not reached!")
        return 0, 0, 10 * tol
    raise NotImplementedError
    return N, S, err
