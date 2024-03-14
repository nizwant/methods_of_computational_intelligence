import unittest
import numpy as np
from ..implementation import activation


class TestActivationFunctions(unittest.TestCase):
    def test_sigmoid(self):
        sigmoid = activation.Sigmoid()
        self.assertAlmostEqual(sigmoid.activation_function(0), 0.5)
        self.assertAlmostEqual(sigmoid.activation_function_derivative(0), 0.25)

    def test_relu(self):
        relu = activation.ReLU()
        self.assertEqual(relu.activation_function(-1), 0)
        self.assertEqual(relu.activation_function_derivative(-1), 0)
        self.assertEqual(relu.activation_function(1), 1)
        self.assertEqual(relu.activation_function_derivative(1), 1)

    def test_tanh(self):
        tanh = activation.Tanh()
        self.assertAlmostEqual(tanh.activation_function(0), 0)
        self.assertAlmostEqual(tanh.activation_function_derivative(0), 1)

    def test_softmax(self):
        softmax = activation.Softmax()
        result = softmax.activation_function(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(np.sum(result), 1)
        self.assertTrue((result >= 0).all() and (result <= 1).all())


if __name__ == "__main__":
    unittest.main()
