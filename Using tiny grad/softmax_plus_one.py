# Inspiration: https://www.evanmiller.org/attention-is-off-by-one.html

# softmax plus one will be defined as a reduce op within Tensor 
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import Context
import numpy as np
Device.DEFAULT = "CPU"
N = 1000
data = [-1000000]*N
logits = Tensor(data)

# Need to account for the overflow hack e^{x_i - max{X}}
# such that 1  + \sum e^{x_i - max{X}} are on the same scale
# 1 -> 1*e^{- max{X}} ??

softmax = logits.softmax()
softmaxp1 = logits.softmaxp1()

with Context(DEBUG=0): 
    print(f"SOFTMAX: {softmax.numpy()[0]}")
    print(f"SOFTMAX PLUS 1: {softmaxp1.numpy()[0]}")