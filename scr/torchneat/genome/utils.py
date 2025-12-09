import torch
from torchneat.common import fetch_first, I_INF
from .gene import BaseGene

def unflatten_conns(nodes: torch.Tensor, conns: torch.Tensor) -> torch.Tensor:
    """
    Transform the (C, CL) connections to (N, N), which contains the idx of the connection in conns.
    Connection length, N means the number of nodes, C means the number of connections.
    Returns the unflattened connection indices with shape (N, N).
    """
    N = nodes.shape[0]  # max_nodes
    C = conns.shape[0]  # max_conns
    node_keys = nodes[:, 0]
    i_keys, o_keys = conns[:, 0], conns[:, 1]

    def key_to_indices(key, keys):
        return fetch_first(key == keys)

    i_idxs = torch.tensor([key_to_indices(key, node_keys) for key in i_keys], dtype=torch.int32)
    o_idxs = torch.tensor([key_to_indices(key, node_keys) for key in o_keys], dtype=torch.int32)

    # Create the unflattened array
    unflatten = torch.full((N, N), I_INF, dtype=torch.int32)
    unflatten[i_idxs, o_idxs] = torch.arange(C, dtype=torch.int32)

    return unflatten


def valid_cnt(nodes_or_conns: torch.Tensor) -> int:
    """
    Count the number of valid (non-NaN) entries in the first column of the tensor.
    """
    return torch.sum(~torch.isnan(nodes_or_conns[:, 0])).item()


def extract_gene_attrs(gene: BaseGene, gene_array: torch.Tensor) -> torch.Tensor:
    """
    Extract the custom attributes of the gene.
    """
    return gene_array[len(gene.fixed_attrs):]


def set_gene_attrs(gene: BaseGene, gene_array: torch.Tensor, attrs: torch.Tensor) -> torch.Tensor:
    """
    Set the custom attributes of the gene.
    """
    gene_array[len(gene.fixed_attrs):] = attrs
    return gene_array


def add_node(nodes: torch.Tensor, fix_attrs: torch.Tensor, custom_attrs: torch.Tensor) -> torch.Tensor:
    """
    Add a new node to the genome.
    The new node will be placed at the first NaN row.
    """
    pos = fetch_first(torch.isnan(nodes[:, 0]))
    nodes[pos] = torch.cat((fix_attrs, custom_attrs))
    return nodes


def delete_node_by_pos(nodes: torch.Tensor, pos: int) -> torch.Tensor:
    """
    Delete a node from the genome.
    Delete the node by its position in nodes.
    """
    nodes[pos] = float('nan')
    return nodes


def add_conn(conns: torch.Tensor, fix_attrs: torch.Tensor, custom_attrs: torch.Tensor) -> torch.Tensor:
    """
    Add a new connection to the genome.
    The new connection will be placed at the first NaN row.
    """
    pos = fetch_first(torch.isnan(conns[:, 0]))
    conns[pos] = torch.cat((fix_attrs, custom_attrs))
    return conns


def delete_conn_by_pos(conns: torch.Tensor, pos: int) -> torch.Tensor:
    """
    Delete a connection from the genome.
    Delete the connection by its index.
    """
    conns[pos] = float('nan')
    return conns


def re_cound_idx(
    nodes: torch.Tensor, conns: torch.Tensor, input_idx: list, output_idx: list
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Make the key of hidden nodes continuous.
    Also update the index of connections.
    """
    next_key = max(*input_idx, *output_idx) + 1
    old2new = {}
    for i, key in enumerate(nodes[:, 0]):
        if torch.isnan(key):
            continue
        if key in input_idx + output_idx:
            continue
        old2new[int(key.item())] = next_key
        next_key += 1

    new_nodes = nodes.clone()
    for i, key in enumerate(nodes[:, 0]):
        if not torch.isnan(key) and int(key.item()) in old2new:
            new_nodes[i, 0] = old2new[int(key.item())]

    new_conns = conns.clone()
    for i, (i_key, o_key) in enumerate(conns[:, :2]):
        if not torch.isnan(i_key) and int(i_key.item()) in old2new:
            new_conns[i, 0] = old2new[int(i_key.item())]
        if not torch.isnan(o_key) and int(o_key.item()) in old2new:
            new_conns[i, 1] = old2new[int(o_key.item())]

    return new_nodes, new_conns

def split_generator(base_gen: torch.Generator, num_splits: int):
    """
    Mimic JAX's random.split by creating independent PyTorch generators.
    Args:
        base_gen: The base torch.Generator instance.
        num_splits: The number of independent generators to create.
    Returns:
        A list of independent torch.Generator instances.
    """
    seeds = torch.randint(0, 2**32, (num_splits,), generator=base_gen)
    return [torch.Generator().manual_seed(seed.item()) for seed in seeds]