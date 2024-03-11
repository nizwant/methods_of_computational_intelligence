import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class NeuralNetwork:
    """
    input_len - długość wektora wejściowego liczb rzeczywistych - domyślnie 1

    output_len - długość wektora wyjściowego liczb rzeczywistych - domyślnie 1

    number_of_layers - liczba naturalna mówiąca ile jest warstw (ukrytych)

    number_of_neurons_per_layer - wektor liczb naturalnych, którego i-ty element mówi ile jest neuronów w i-tej warstwie (ukrytej)

    weights - ciąg macierzy wag dla każdego których ij element macierzy mówi o wadze przejścia z i-tego neuronu do j-tego w kolejnej warstwie
    : 1D-ndarray[2D-ndarray[float]]
        macierz 0 - macierz wag pomiędzy warstwą zero (inputem) a warstwą pierwszą;
        macierz wag wyjściowych dla zerowej warstwy a wejściowych dla pierwszej warstwy,

    biases - ciąg wektorów biasów, których i element mówi o tym jak wygląda bias w danej warstwie, w i-tym neuronie
    : 1D-ndarray[1D-ndarray[float]]
        wektor 0 - wektor biasów dla warstwy pierwszej

    neurons - ciąg wektorów neuronów, których i-ta wartość to neuron i-ty neuron w konkretnej warstwie
    """

    input_len = 1
    output_len = 1

    def __init__(self, number_of_layers, number_of_neurons_per_layer, weights, biases):
        self.number_of_layers = number_of_layers + 1  # + 1 bo dodajemy output
        self.number_of_neurons_per_layer = number_of_neurons_per_layer

        # self.number_of_neurons_per_layer.insert(0, self.input_len)
        self.number_of_neurons_per_layer.append(self.output_len)

        if not len(self.number_of_neurons_per_layer) == self.number_of_layers:
            raise ValueError(
                "Number of layers and number of neurons per layer LENGTH error"
            )

        self.weights = weights
        self.biases = biases

        for i in range(self.number_of_layers):
            if not self.weights[i].shape[1] == self.number_of_neurons_per_layer[i]:
                raise ValueError(
                    f"Weights error between layers {i} and {i+1} - number of columns"
                )
            if not self.weights[i].shape[0] == self.number_of_neurons_per_layer[i - 1]:
                raise ValueError(
                    f"Weights error between layers {i} and {i+1} - number of rows"
                )
            if not np.size(self.biases[i]) == self.number_of_neurons_per_layer[i]:
                raise ValueError(f"Biases error on layer {i+1}")

        self.neurons = [[Neuron()] * N for N in self.number_of_neurons_per_layer]
        # zmiana funkcji aktywacji w "neuronach" wyjściowych
        self.change_activation_function_in_layer(self.number_of_layers - 1, lambda x: x)

        self.z = None
        self.a = None
        self.errors = None
        self.weights_gradient = [np.zeros_like(weigh) for weigh in weights]
        self.biases_gradient = [np.zeros_like(bias) for bias in biases]
        self.MSE = []

    def __calculate_activation(self, input, layer_index, verbose=False):
        if verbose:
            print("__________________________")
            print(f"Layer {layer_index+1}:")
            print("Weights:", self.weights[layer_index])
            print("Biases:", self.biases[layer_index])
            print("Layer input shape:", input.shape)
            print("Layer input:", input)

        layer_output = (
            np.matmul(input, self.weights[layer_index]) + self.biases[layer_index]
        )
        self.z.append(layer_output)
        if verbose:
            print(
                "Layer output after matrix multiplication and adding biases:",
                layer_output.shape,
            )
            print("Layer output:", layer_output)

        layer_output = np.array(
            [
                [
                    self.neurons[layer_index][index].function(sum)
                    for sum, index in zip(
                        layer_output[0, :], np.arange(layer_output.shape[1])
                    )
                ]
            ]
        )

        if verbose:
            print("Layer output after activation function:", layer_output)

        return layer_output

    def __feed_forward(self, input, verbose=False):

        # wejście - obliczanie sigm w pierwszej warstwie i zastosowanie funkcji aktywacji w nich
        i = 0
        self.z = [input]
        self.a = [input]
        layer_output = self.__calculate_activation(input, 0, verbose=verbose)
        self.a.append(layer_output)

        # warstwy ukryte - obliczanie sigm w kolejnych warstwach ukrytych i zastosowanie funkcji aktywacji w nich

        for i in range(
            1, self.number_of_layers
        ):  # dla każdej warstwy ukrytej mnożenie przez wagi i dodanie biasu
            layer_output = self.__calculate_activation(layer_output, i, verbose=verbose)
            self.a.append(layer_output)

        return layer_output

    def back_propagation(self, desired_y, verbose=False):

        # works only for sigmoid activation function

        self.weights_gradient = []
        self.biases_gradient = []

        # 0. Forward feed - must do it before backpropagation
        # 1. Error on output layer
        z_output = self.z[-1]
        error = z_output - desired_y
        # error = np.subtract(z_output, desired_y)
        # sigmoid_derivative_output = sigmoid_derivative(z_output)
        # error = np.multiply(error, sigmoid_derivative_output)
        self.errors = [error]

        if verbose:
            print(f"z: {z_output} | shape:{z_output.shape}")
            print(f"error: {error} | shape:{error.shape}")

        # 2. Backward feed
        for i in range(self.number_of_layers - 1, -1, -1):
            error = np.matmul(error, self.weights[i].T)
            # TU SPRAWDZIC Z
            sigmoid_derivative_output = sigmoid_derivative(self.z[i])
            error = np.multiply(error, sigmoid_derivative_output)
            self.errors.insert(0, error)
            # np.insert(self.errors, 0, error)

            # 3. Update weights and biases
            if verbose:
                print(f"z: {self.z[i-1]} | shape:{self.z[i-1].shape}")
                print(f"a: {self.a[i-1]} | shape:{self.a[i-1].shape}")
                print(f"error: {error} | shape:{error.shape}")
            if len(self.weights_gradient) != len(self.weights) or len(
                self.biases_gradient
            ) != len(self.biases):
                weights_gradient = np.matmul(self.a[i - 1].T, error)
                self.weights_gradient.insert(0, weights_gradient)
                biases_gradient = error
                self.biases_gradient.insert(0, error)
            else:
                weights_gradient = np.matmul(self.a[i - 1].T, error)
                self.weights_gradient[i] += weights_gradient
                biases_gradient = error
                self.biases_gradient[i] += biases_gradient

    def gradient_descent(
        self, input, desired_output, batch_size, learning_rate=0.01, verbose=False
    ):
        num_batches = len(input) // batch_size
        if len(input) % batch_size != 0:
            num_batches += 1
        x_batches = np.array_split(input, num_batches)
        y_batches = np.array_split(desired_output, num_batches)
        predicted_output = []
        for x_batch, y_batch in zip(x_batches, y_batches):
            if (
                not type(x_batch) == list
                and not type(x_batch) == np.ndarray
                and not type(x_batch) == pd.Series
            ):
                x_batch = np.array([[x_batch]])
            for x, y in zip(x_batch, y_batch):
                x = np.array([[x]])
                self.__feed_forward(x, verbose=verbose)
                self.back_propagation(y, verbose=verbose)
                for i in range(self.number_of_layers - 1):
                    self.weights[i] = np.subtract(
                        self.weights[i],
                        learning_rate / len(x_batch) * self.weights_gradient[i],
                        casting="safe",
                    )
                    self.biases[i] = np.subtract(
                        self.biases[i],
                        learning_rate / len(x_batch) * self.biases_gradient[i],
                        casting="safe",
                    )
                predicted_output.append(self.a[-1][0][0])
        predicted_output = np.array(predicted_output)
        if verbose:
            print(self.weights_gradient)
            print(self.biases_gradient)
        # print(f'predicted_output shape: {predicted_output.shape}')
        # print(f'desired_output shape: {desired_output.to_numpy().shape}')
        self.MSE.append(mean_squared_error(desired_output.to_numpy(), predicted_output))

    def fit(
        self,
        input,
        desired_output,
        epochs=100,
        learning_rate=0.01,
        batch_size=10,
        verbose=False,
    ):
        for epoch in range(epochs):
            # input = np.random.permutation(input)

            self.gradient_descent(
                input, desired_output, batch_size, learning_rate, verbose
            )
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"MSE: {self.MSE[-1]}")

    def predict(self, input, verbose=False):
        """
        Funkcja która zwraca przewidywany wektor wyjsciowy (lub liczbę w przypadku output_len = 1 )
        """
        if not type(input) == list and not type(input) == np.ndarray:
            input = np.array([[input]])

        self.input = input
        # 1. Chceck if input has correct dimension
        if not len(self.input) == self.input_len:
            raise ValueError("Input lenght error")
        # 2. Forward feed
        output = self.__feed_forward(input, verbose)
        if output.shape[1] == 1 and output.shape[0] == 1:
            return output[0][0]
        else:
            return output

    def change_input_len(self, new_input_len):
        self.input_len = new_input_len

    def change_output_len(self, new_output_len):
        self.output_len = new_output_len

    def change_number_of_layers(self, new_number_of_layers):
        self.number_of_layers = new_number_of_layers

    def change_number_of_neurons_per_layer(self, new_number_of_neurons_per_layer):
        self.number_of_neurons_per_layer = new_number_of_neurons_per_layer

    def change_weights(self, new_weights):
        self.weights = new_weights

    def change_biases(self, new_biases):
        self.biases = new_biases

    # zmien funkcje aktywacji w każdym neuronie
    def change_activation_function(self, new_activation_function):
        for neuron in self.neurons:
            neuron.function = new_activation_function

    # zmien funkcje aktywacji w konkretnym neuronie
    def change_activation_function_in_neuron(
        self, neuron_index, layer_index, new_activation_function
    ):
        # layer_index = layer_index - 1 #startujemy z 0 - warstwa pierwsza to warstwa ukryta zerowa
        self.neurons[layer_index][neuron_index].function = new_activation_function

    # zmien funkcje aktywacji w konkretnej warstwie
    def change_activation_function_in_layer(self, layer_index, new_activation_function):
        # layer_index = layer_index - 1 #startujemy z 0 - warstwa pierwsza to warstwa ukryta zerowa
        for i in range(self.number_of_neurons_per_layer[layer_index]):
            self.neurons[layer_index][i].function = new_activation_function

    def mean_squared_error(self, input, output):
        """
        Calculate the mean squared error of the network on a given dataset and output
        """
        mse = []
        for i, j in zip(input, output):
            mse.append((j - self.predict(i)) ** 2)
        return np.mean(mse)

    def calculate_gradient_numeracly(self, input, output):
        """
        Calculate the gradient of the network
        """
        h = 0.0000001
        original_mse = self.mean_squared_error(input, output)

        for layer_number, weight in enumerate(self.weights):
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    self.weights[layer_number][i, j] += h
                    new_mse = self.mean_squared_error(input, output)
                    self.weights_gradient[layer_number][i, j] = (
                        new_mse - original_mse
                    ) / h
                    self.weights[layer_number][i, j] -= h

        for layer_number, bias in enumerate(self.biases):
            for i in range(bias.shape[0]):
                for j in range(bias.shape[1]):
                    self.biases[layer_number][i, j] += h
                    new_mse = self.mean_squared_error(input, output)
                    self.biases_gradient[layer_number][i, j] = (
                        new_mse - original_mse
                    ) / h
                    self.biases[layer_number][i, j] -= h


class Neuron:
    def __init__(self):
        self.function = sigmoid


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    # return np.exp(x)/(1+np.exp(x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def logit(x):
    return np.log(x / (1 - x))


if __name__ == "__main__":

    # weights = np.array( [np.array([[1,1,-1,-1,1]]), np.array([[1,-1,-1,-1,1]]).reshape(5,1) ], dtype=np.ndarray)
    # biases = np.array( [[1,0,0,0,0], [1]], dtype=object)
    # NN = NeuralNetwork(1, [5], weights, biases)
    # print(NN.predict(0.8))

    weights = np.array(
        [
            np.array([[1, 2, -1, -0.5, 1]]),
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1],
                ]
            ),
            np.array([[1, -1, -1, -1, 1]]).reshape(5, 1),
        ],
        dtype=np.ndarray,
    )
    biases = np.array([[1, 0, 0, 0, 0], [1, -1, 0, 0, 0], [0]], dtype=object)
    NN = NeuralNetwork(2, [5, 5], weights, biases)
    # print(NN.predict(0.8))
    # NN.back_propagation( 0.8, verbose=True)

    print(NN.weights)
    NN.gradient_descent([0.8, 0.7], [0.8, 0.6], batch_size=10, learning_rate=0.01)
    print(NN.weights)
