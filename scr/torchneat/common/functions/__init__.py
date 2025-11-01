from .act_torch import *

act_name2torch = {
    "scaled_sigmoid": scaled_sigmoid_,
    "sigmoid": sigmoid_,
    "scaled_tanh": scaled_tanh_,
    "tanh": tanh_,
    "sin": sin_,
    "relu": relu_,
    "lelu": lelu_,
    "identity": identity_,
    "inv": inv_,
    "log": log_,
    "exp": exp_,
    "abs": abs_
}