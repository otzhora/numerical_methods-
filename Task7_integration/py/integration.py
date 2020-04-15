import numpy as np

from scipy.special import comb
from scipy import integrate

import inspect

MAXITER = 12


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

    for j in range(max_s + 1):
        if beta == 0:
            coefs = np.array([comb(j, i) * a ** i / (j - i - alpha + 1)
                              for i in range(j + 1)])
            r = np.array([(xr - a) ** (j - i - alpha + 1)
                          for i in range(j + 1)])
            l = np.array([(xl - a) ** (j - i - alpha + 1)
                          for i in range(j + 1)])

        if alpha == 0:
            coefs = np.array([(-1) ** (j-i) * comb(j, i) * b ** i / (j - i - beta + 1)
                              for i in range(j + 1)])
            r = np.array([(b - xl) ** (j - i - beta + 1)
                          for i in range(j + 1)])
            l = np.array([(b - xr) ** (j - i - beta + 1)
                          for i in range(j + 1)])

        mu[j] = sum(coefs * (r - l))
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
    n = len(nodes)

    X = [[nodes[i] ** s for i in range(n)] for s in range(n)]
    A = np.linalg.solve(X, mu)

    return [f(nodes[i]) for i in range(len(nodes))] @ A


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

    X = np.array([[mu[j + s] for j in range(n)] for s in range(n)])

    coefs = np.linalg.solve(X, [-mu[n + s] for s in range(n)])
    coefs = coefs[::-1]
    coefs = np.concatenate([1., coefs], axis=None)

    nodes = np.roots(coefs)

    X = [[nodes[i] ** s for i in range(n)] for s in range(n)]
    A = np.linalg.solve(X, mu[:n])

    return [f(nodes[i]) for i in range(len(nodes))] @ A


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
    return abs(s2 - s1) / (L ** m - 1)
    # return abs((s2 - s1) / (1 - L ** (-m)))


def aitken(s1: float, s2: float, s3: float, L: float):
    """
    estimate convergence order
    s1, s2, s3: consecutive composite quads
    return: convergence order estimation
    """

    return -np.log(abs(s3 - s2) / abs(s2 - s1)) / np.log(L)


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

    iter = 0
    N = 8

    s1 = composite_quad(f, xl, xr, N, n, *params)
    s2 = composite_quad(f, xl, xr, 2 * N, n, *params)

    while iter < MAXITER:
        err = runge(s1, s2, 2, 3)
        if err <= tol:
            return 2 * N, s2, err

        N *= 2
        s1 = s2
        s2 = composite_quad(f, xl, xr, 2 * N, n, *params)
        iter += 1
    print("Convergence not reached!")
    return 0, 0, 10*tol


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

    iter = 0
    N = 8

    s1 = composite_quad(f, xl, xr, N, n, *params)
    s2 = composite_quad(f, xl, xr, 2 * N, n, *params)
    s3 = composite_quad(f, xl, xr, 4 * N, n, *params)

    m = aitken(s1, s2, s3, 2)
    while iter < MAXITER:
        err = runge(s1, s2, 2, m)
        if err <= tol:
            return 2 * N, s2, err, m

        N *= 2

        s1 = s2
        s2 = s3
        s3 = composite_quad(f, xl, xr, 4 * N, n, *params)

        m = aitken(s1, s2, s3, 2)
        iter += 1

    print("Convergence not reached!")
    return 0, 0, 10*tol, 0


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

    iter = 0
    N = 8

    s1 = composite_gauss(f, xl, xr, N, n, *params)
    s2 = composite_gauss(f, xl, xr, 2 * N, n, *params)

    while iter < MAXITER:
        err = runge(s1, s2, 2, 3)
        if err <= tol:
            return 2 * N, s2, err

        N *= 2
        s1 = s2
        s2 = composite_gauss(f, xl, xr, 2 * N, n, *params)

    print("Convergence not reached!")
    return 0, 0, 10*tol


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

    iter = 0
    N = 8

    s1 = composite_gauss(f, xl, xr, N, n, *params)
    s2 = composite_gauss(f, xl, xr, 2 * N, n, *params)
    s3 = composite_gauss(f, xl, xr, 4 * N, n, *params)

    m = aitken(s1, s2, s3, 2)
    while iter < MAXITER:
        err = runge(s1, s2, 2, m)
        if s3 - s2 > 0 or s2 - s1 > 0:
            print("convergence not reached yet", f"err: {err}")
        if err <= tol:
            return 2 * N, s2, err, m

        N *= 2

        s1 = s2
        s2 = s3
        s3 = composite_gauss(f, xl, xr, 4 * N, n, *params)

        m = aitken(s1, s2, s3, 2)

    print("Convergence not reached!")
    return 0, 0, 10*tol, 0


def optimal_nc(f, xl: float, xr: float, n: int, tol: float, *params):
    """ estimate the optimal step with Aitken and Runge procedures
    f : function to integrate
    xl : left limit
    xr : right limit
    n : nodes number in the small formula
    tol : error tolerance
    *params : arguments to pass to composite_quad function
    """

    N = 4

    s1 = composite_quad(f, xl, xr, N, n, *params)
    s2 = composite_quad(f, xl, xr, N * 2, n, *params)
    s3 = composite_quad(f, xl, xr, N * 4, n, *params)
    m = aitken(s1, s2, s3, 2)

    h = (xr - xl) / N
    R = runge(s2, s3, 2, m)
    hopt = 0.85 * h * np.sqrt(tol / R)
    N = int(np.ceil((xr - xl) / hopt))

    s1 = composite_quad(f, xl, xr, N, n, *params)
    s2 = composite_quad(f, xl, xr, N * 2, n, *params)
    s3 = composite_quad(f, xl, xr, N * 4, n, *params)

    m = aitken(s1, s2, s3, 2)
    err = runge(s1, s2, 2, m)

    return N, s2, err
