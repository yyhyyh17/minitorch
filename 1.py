from numba import njit, prange
import numpy as np
import time
import numba
import functools
numba.config.NUMBA_DEFAULT_NUM_THREADS=10

a = np.random.randn(100,1024,1024)
b =  np.random.randn(100,1024,1024)

i, j, k=a.shape

def get_runtime(func):
    def wrapper(*val):
        time1 = time.time()
        func(*val)
        print(func.__name__, 'time: ', time.time() - time1)
    return wrapper 

@get_runtime
@njit(parallel=True)
def do_trig2(x, y):
    z = np.empty_like(x)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            z[i, j] = np.sin(x[i, j]**2) + np.cos(y[i, j])
    return z

@get_runtime
@njit
def do_trig(x, y):
    z = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = np.sin(x[i, j]**2) + np.cos(y[i, j])
    return z

x = np.random.random((1000, 1000))
y = np.random.random((1000, 1000))


do_trig(x, y)

do_trig2(x, y)
import minitorch
t = minitorch.Scalar(2)
s = minitorch.Scalar(3)
k = t + s
k.backward()
