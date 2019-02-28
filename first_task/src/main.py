import math
import numpy as np

def sqrt(n, eps=1e-15):
    x = 1
    while(True):
        nx = (x + n / x) / 2;
        if abs(x - nx) < eps:
            break
        x = nx
    return x


def ch(n, eps=1e-6):
    ans = 0
    k = 0
    while(True):
        u = n ** k / math.factorial(k)
        if u < eps:
            break
        ans += u
        k += 2
    return ans


def cos(n, eps=1e-6):
    ans = 0
    k = 0
    while(True):
        u = n ** (2 * k) / math.factorial(2 * k)
        if u < eps:
            break
        ans += ((-1) ** k) *u
        k += 1

    return ans


def f(x, eps=1e-6):
    return ch(1 + sqrt(1 + x), eps=eps / 1.8) * cos(sqrt(1 + x - x ** 2), eps=eps / 12.6)


def math_f(x):
    return math.cosh(1 + math.sqrt(1 + x)) * math.cos(math.sqrt(1 + x - x ** 2))



print('f values')
for x in np.arange(0.1, 0.21, 0.01):
    print('{} & {} \\\\'.format(x, f(x)))

print('math_f values')
for x in np.arange(0.1, 0.21, 0.01):
    print('{} & {} \\\\'.format(x, math_f(x)))

print('max diff')
res = -1
for x in np.arange(0.1, 0.21, 0.01):
    res = max(res, abs(math_f(x) - f(x)))
print(res)
""" x = 0.1
print('sqrt(x): ', sqrt(x))
print('x ** 0.5: ', x ** 0.5)
print('math.sqrt(x)', math.sqrt(x))

print()

print('ch(x)', ch(x))
print('math.cosh(x)', math.cosh(x))

print()

print('cos(x)', cos(x))
print('math.cos(x)', math.cos(x))

print()

print('f(x)', f(x))
print('math_f(x)', math_f(x))
print('f(x) - math_f(x)', abs(f(x) - math_f(x))) """