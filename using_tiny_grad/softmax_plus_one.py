# Inspiration: https://www.evanmiller.org/attention-is-off-by-one.html

# softmax plus one will be defined as a reduce op within Tensor 
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import Context
import numpy as np
Device.DEFAULT = "CPU"

# Need to account for the overflow hack e^{x_i - max{X}}
# such that 1  + \sum e^{x_i - max{X}} are on the same scale
# 1 -> 1*e^{- max{X}} ??

# Numpy 
def _softmax(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    n = x - m
    e = np.exp(n)
    return n, e, np.sum(e, axis=axis, keepdims=True), m

def np_softmaxplus1(x, axis=-1):
    _, e, ss, m = _softmax(x, axis=axis)
    denom = np.exp(-m) + ss
    return np.divide(e, denom)

def np_softmax(x, axis=-1):
    _, e, ss, _ = _softmax(x, axis=axis)
    return np.divide(e, ss)

# Implement as reduction op in tinygrad.Tensor

def test_softmaxplus1():
    pass


if __name__ == "__main__": 
    N = 4
    t1 = np.array([10]*N)
    t2 = np.array([-10]*N)
    t3 = np.array([-10, 0, 10])

    for test in [t1, t2, t3]:
        np_sm = np_softmax(test)
        np_smp1 = np_softmaxplus1(test)
        print(f"In: {test}")
        print(f"SOFTMAX: {sm}")
        print(f"SOFTMAX PLUS 1: {smp1}")

    
