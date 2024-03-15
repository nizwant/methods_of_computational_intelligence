from cost_function import MeanSquaredError
from cost_function import AbsoluteError


class CostFunctionBuilder:
    def build_cost_function(self, cost_function: str):
        if cost_function == "mse":
            return MeanSquaredError()
        elif cost_function == "ae":
            return AbsoluteError()
        else:
            raise ValueError("Invalid cost function")
