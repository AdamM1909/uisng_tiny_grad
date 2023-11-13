from __future__ import annotations
from typing import Optional, Tuple, Union, Any, Dict, Callable, Type, List, ClassVar
from enum import Enum, auto
from abc import ABC

# we will be using the clang backend
from tinygrad.ops import Device
Device.DEFAULT = "GPU"

# first, 2+3 as a Tensor, the highest level
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context
# a = Tensor([2])
# b = Tensor([3])
N = 1024
# a, b = Tensor.rand(N, N), Tensor.rand(N, N)
result = a + b
with Context(DEBUG=3): 
    print(f"{a.numpy()} + {b.numpy()} = {result.numpy()}")
# assert result.numpy()[0] == 5.
