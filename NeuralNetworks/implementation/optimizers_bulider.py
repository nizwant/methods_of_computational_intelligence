from optimizers import mini_batch_gradient_descent
from optimizers import stochastic_gradient_descent
from optimizers import full_batch_gradient_descent
from optimizers import mini_batch_gradient_descent_with_momentum
from optimizers import adagrad
from optimizers import rmsprop
from optimizers import adam


class OptimizersBuilder:
    def __init__(self):
        pass

    def build_optimizer(self, optimizer_name):
        if optimizer_name == "mini_batch_gradient_descent":
            return mini_batch_gradient_descent.optimize
        elif optimizer_name == "stochastic_gradient_descent":
            return stochastic_gradient_descent.optimize
        elif optimizer_name == "full_batch_gradient_descent":
            return full_batch_gradient_descent.optimize
        elif optimizer_name == "mini_batch_gradient_descent_with_momentum":
            return mini_batch_gradient_descent_with_momentum.optimize
        elif optimizer_name == "adagrad":
            return adagrad.optimize
        elif optimizer_name == "rmsprop":
            return rmsprop.optimize
        elif optimizer_name == "adam":
            return adam.optimize
        else:
            raise ValueError("Optimizer name is not supported")
