from typing import Tuple, Set, List, Union
import torch


def topological_sort(nodes: torch.tensor, conns: torch.tensor) -> torch.tensor:
    """
    A PyTorch version of topological_sort.
    Args:
        nodes: Tensor of shape [N, ...] representing the nodes.
        conns: Tensor of shape [N, N] representing the adjacency matrix of connections.
    Returns:
        A tensor representing the topological order of the nodes.
    """
    # Compute in-degree for each node
    in_degree = torch.where(
        torch.isnan(nodes[:, 0]), 
        float('nan'), 
        torch.sum(conns, dim=0)
    )
    res = torch.full_like(in_degree, float('inf'))

    idx = 0

    while True:
        # Find the first node with in-degree 0
        zero_in_degree = (in_degree == 0).nonzero(as_tuple=True)[0]
        if zero_in_degree.numel() == 0:
            break

        i = zero_in_degree[0].item()

        # Add to result and mark as visited
        res[idx] = i
        idx += 1
        in_degree[i] = -1

        # Decrease in-degree of all its children
        children = conns[i, :]
        in_degree = torch.where(children, in_degree - 1, in_degree)

    return res

def check_cycles(nodes: torch.Tensor, conns: torch.Tensor, from_idx: int, to_idx: int) -> bool:
    """
    Check whether adding a new connection (from_idx -> to_idx) will cause a cycle.
    Args:
        nodes: Tensor of shape [N, ...] representing the nodes.
        conns: Tensor of shape [N, N] representing the adjacency matrix of connections.
        from_idx: Index of the source node.
        to_idx: Index of the target node.
    Returns:
        A boolean indicating whether the new connection creates a cycle.
    """
    # Add the new connection
    conns[from_idx, to_idx] = True

    visited = torch.full((nodes.shape[0],), False, dtype=torch.bool)
    new_visited = visited.clone()
    new_visited[to_idx] = True

    while True:
        # Check if no new nodes are visited or if the starting node is visited
        if torch.equal(visited, new_visited) or new_visited[from_idx]:
            break

        visited = new_visited.clone()
        new_visited = torch.matmul(visited.float(), conns).bool()
        new_visited = torch.logical_or(visited, new_visited)

    return new_visited[from_idx].item()