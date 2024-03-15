import unittest
import numpy as np
from NeuralNetworks.code.implementation import activation


class TestActivationFunctions(unittest.TestCase):
    def test_sigmoid(self):
        sigmoid = activation.Sigmoid()
        self.assertAlmostEqual(sigmoid.activation(0), 0.5)
        self.assertAlmostEqual(sigmoid.derivative(0), 0.25)

    def test_relu(self):
        relu = activation.ReLU()
        self.assertEqual(relu.activation(-1), 0)
        self.assertEqual(relu.derivative(-1), 0)
        self.assertEqual(relu.activation(1), 1)
        self.assertEqual(relu.derivative(1), 1)

    def test_tanh(self):
        tanh = activation.Tanh()
        self.assertAlmostEqual(tanh.activation(0), 0)
        self.assertAlmostEqual(tanh.derivative(0), 1)

    def test_softmax(self):
        softmax = activation.Softmax()
        result = softmax.activation(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(np.sum(result), 1)
        self.assertTrue((result >= 0).all() and (result <= 1).all())


if __name__ == "__main__":
    unittest.main()


# sigmoid = Sigmoid()
# print(sigmoid.activation(np.array([1.0, 2.0, 3.0])))
# print(sigmoid.derivative (np.array([1.0, 2.0, 3.0])))

# print(sigmoid.activation(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])))
# print(
#     sigmoid.derivative (np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
# )
# print()

# relu = ReLU()
# print(relu.activation(np.array([1.0, -2.0, 3.0])))
# print(relu.derivative (np.array([1.0, -2.0, 3.0])))

# print(relu.activation(np.array([[-1.0, 2.0, 3.0], [1.0, -2.0, 3.0]])))
# print(
#     relu.derivative (np.array([[-1.0, 2.0, 3.0], [1.0, -2.0, 3.0]]))
# )
# print()

# tanh = Tanh()
# print(tanh.activation(np.array([1.0, 2.0, 3.0])))
# print(tanh.derivative (np.array([1.0, 2.0, 3.0])))
# print(tanh.activation(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])))
# print(tanh.derivative (np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])))
# print()

# softmax = Softmax()
# print(softmax.activation(np.array([-1.0, 2.0, 3.0])))
# print(softmax.derivative (np.array([-1.0, 2.0, 3.0])))
# print(softmax.activation(np.array([[1.0, 2.0, 6.0], [-11.0, 2.0, 3.0]])))
# print(
#     softmax.derivative (np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
# )
# print()

# linear = Linear()
# print(linear.activation(np.array([1.0, 2.0, 3.0])))
# print(linear.derivative (np.array([1.0, 2.0, 3.0])))
# print(linear.activation(np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])))
# print(
#     linear.derivative (np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
# )
# print()

# leaky_relu = LeakyReLU()
# print(leaky_relu.activation(np.array([1.0, -2.0, 3.0])))
# print(leaky_relu.derivative (np.array([1.0, 2.0, -3.0])))
# print(leaky_relu.activation(np.array([[-1.0, 2.0, 3.0], [1.0, 2.0, -3.0]])))
# print(
#     leaky_relu.derivative (
#         np.array([[1.0, 2.0, 3.0], [1.0, -2.0, 3.0]])
#     )
# )
# print()
