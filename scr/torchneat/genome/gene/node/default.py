from typing import Optional, Union, Sequence, Callable
from .base import BaseNode
import torchneat.common.functions.act_torch as torchneat_act

class DefaultNode(BaseNode):
    "Default node gene, with the same behavior as in NEAT-python."

    custom_attrs = ["bias", "response", "aggregation", "activation"]

    def __init__(
        self,
        bias_init_mean: float = 0.0,
        bias_init_std: float = 1.0,
        bias_mutate_power: float = 0.15,
        bias_mutate_rate: float = 0.2,
        bias_replace_rate: float = 0.015,
        bias_lower_bound: float = -5,
        bias_upper_bound: float = 5,
        response_init_mean: float = 1.0,
        response_init_std: float = 0.0,
        response_mutate_power: float = 0.15,
        response_mutate_rate: float = 0.2,
        response_replace_rate: float = 0.015,
        response_lower_bound: float = -5,
        response_upper_bound: float = 5,
        aggregation_default: Optional[Callable] = None,
        aggregation_options: Union[Callable, Sequence[Callable]] = torchneat_act.sum,
        aggregation_replace_rate: float = 0.1,
        activation_default: Optional[Callable] = None,
        activation_options: Union[Callable, Sequence[Callable]] = torchneat_act.sigmoid,
        activation_replace_rate: float = 0.1,
    ):
        super().__init__()
