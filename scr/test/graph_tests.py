from torchneat.common.graph import topological_sort, check_cycles
import torch

if __name__ == "__main__":
    # Example nodes and connections
    nodes = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    conns = torch.tensor([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
    ], dtype=torch.bool)

    # Test topological_sort
    topo_order = topological_sort(nodes, conns)
    print("Topological Order:", topo_order)

    # Test check_cycles
    has_cycle = check_cycles(nodes, conns, 3, 0)
    print("Has Cycle:", has_cycle)