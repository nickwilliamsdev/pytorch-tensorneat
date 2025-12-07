import torch

class BasePopulation:
    def __init__(
        self,
        size,
        num_inputs,
        num_outputs,
        layer_indices,
        max_nodes,
        max_conns,
        node_gene,
        conn_gene,
        mutation,
        crossover,
        distance,
        output_transform=None,
        input_transform=None,
    ):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.max_nodes = max_nodes
        self.max_conns = max_conns
        self.node_gene = node_gene
        self.conn_gene = conn_gene
        self.mutation = mutation
        self.crossover = crossover
        self.distance = distance
        self.output_transform = output_transform
        self.input_transform = input_transform

        self.input_idx = torch.tensor(layer_indices[0])
        self.output_idx = torch.tensor(layer_indices[-1])
        self.size = size

    def transform(self, state, nodes, conns):
        raise NotImplementedError

    def forward(self, state, transformed, inputs):
        raise NotImplementedError

    def sympy_func(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError