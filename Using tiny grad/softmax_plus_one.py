# Inspiration: https://www.evanmiller.org/attention-is-off-by-one.html

# softmax plus one will be defined as a reduce op within Tensor 
from tinygrad.tensor import Tensor
from tinygrad.ops import Device
from tinygrad.helpers import Context
Device.DEFAULT = "CPU"

data = [-100, -100, -100, -100]
logits = Tensor(data)

softmax = logits.softmax()
softmaxp1 = logits.softmaxp1()

with Context(DEBUG=2): 
    print(f"SOFTMAX: {softmax.numpy()}")
    print(f"SOFTMAX PLUS 1: {softmaxp1.numpy()}")