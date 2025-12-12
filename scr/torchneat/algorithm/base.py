
class BaseAlgorithm(object):
    def ask(self):
        """require the population to be evaluated"""
        raise NotImplementedError

    def tell(self, fitness):
        """update the state of the algorithm"""
        raise NotImplementedError

    def transform(self, individual):
        """transform the genome into a neural network"""
        raise NotImplementedError

    def forward(self, transformed, inputs):
        raise NotImplementedError

    def show_details(self, fitness):
        """Visualize the running details of the algorithm"""
        raise NotImplementedError

    @property
    def num_inputs(self):
        raise NotImplementedError

    @property
    def num_outputs(self):
        raise NotImplementedError
