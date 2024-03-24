from cost_function import MeanSquaredError
from cost_function import AbsoluteError
from cost_function import CrossEntropyWithSoftmax


class CostFunctionBuilder:
    def build_cost_function(self, cost_function: str):
        if cost_function == "mse":
            return MeanSquaredError()
        elif cost_function == "ae":
            return AbsoluteError()
        elif cost_function == "cross_entropy_with_softmax":
            return CrossEntropyWithSoftmax()
        else:
            raise ValueError("Invalid cost function")
