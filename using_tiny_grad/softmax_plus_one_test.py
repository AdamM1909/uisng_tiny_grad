import unittest
import numpy as np
from tinygrad.tensor import Tensor
from using_tiny_grad.softmax_plus_one import np_softmaxplus1, np_softmax


class TestTinyGradSoftmaxPlusOne(unittest.TestCase):
    def testbase(self):
        x = np.array([1.]*10)
        x_tensor = Tensor(x)
        tiny_grad_smp1 = x_tensor.softmaxp1().numpy()
        np_smp1 = np_softmaxplus1(x)
        np.testing.assert_array_equal(tiny_grad_smp1, np_smp1)

    def testlimit_x_small(self):
        """
        As x_0, x_1, ... -> -inf, e^{x_i}/1 + sum(e^{x_i}) -> 0.

        This is the desrable behaviour from softmaxplus1 i.e. 
        No probability mass on any class if logits are small. 
        """
        x = np.array([-100.]*10)
        x_tensor = Tensor(x)
        tiny_grad_smp1 = x_tensor.softmaxp1().numpy()
        expected = np.array([0.]*10)
        np.testing.assert_almost_equal(tiny_grad_smp1, expected, decimal = 5) 

    def testlimit_x_large(self):
        """
        As x_0, x_1, ... -> inf, e^{x_i}/1 + sum(e^{x_i}) -> e^{x_i}/ sum(e^{x_i})

        I.e. softamxplus1 -> softamx.
        """
        x = np.array([100.]*10)
        x_tensor = Tensor(x)
        tiny_grad_smp1 = x_tensor.softmaxp1().numpy()
        expected =x_tensor.softmax().numpy()
        np.testing.assert_almost_equal(tiny_grad_smp1, expected, decimal = 5) 

 
unittest.main()
