from .base import BaseConn

class DefaultConn(BaseConn):
    "Default connection gene, with the same behavior as in NEAT-python."

    custom_attrs = ["weight"]
    def __init__(self):
        super(DefaultConn, self).__init__()