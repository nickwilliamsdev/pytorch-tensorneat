from ..base import BaseGene


class ConnectionGene(BaseGene):
    "Base class for connection genes."
    fixed_attrs = ["input_index", "output_index"]
    def __init__(self):
        super(ConnectionGene, self).__init__()