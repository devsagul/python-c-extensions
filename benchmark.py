import time
import numpy as np
import helloModule as hm
import itertools
import math as m
import cProfile

def recovery_factor_naive(X):
    maxdrawdown = 0
    peak = float("-inf")
    for x in X:
        peak = max(peak, x)
        maxdrawdown = max(maxdrawdown, peak - x)
    return x / maxdrawdown

def recovery_factor_advanced(X):
    maxdrawdown = max(
                      map(
                          lambda x: x[0] - x[1],
                          zip(
                              itertools.accumulate(X, max),
                              X
                          )
                      )
                  )
    return X[-1] / maxdrawdown

def recovery_factor_numpy(X):
    maxdrawdown = np.max(np.maximum.accumulate(X) - X)
    return X[-1] / maxdrawdown

def recovery_factor_c(X):
    return X[-1] / hm.maxdrawdown(X)

if __name__ == "__main__":
    cProfile.run("""
for _ in range(10 ** 5):
    X = np.cumsum(np.random.normal(loc=0.01, size=10 ** 3))
    a = recovery_factor_naive(X.copy())
    b = recovery_factor_advanced(X.copy())
    c = recovery_factor_numpy(X.copy())
    d = recovery_factor_c(X.copy())
    assert m.isclose(a, b, abs_tol=0.0001)
    assert m.isclose(a, c, abs_tol=0.0001)
    assert m.isclose(a, d, abs_tol=0.0001)
    """)