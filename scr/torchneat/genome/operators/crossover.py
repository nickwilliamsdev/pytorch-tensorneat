import torch
from torch import Tensor
from typing import Tuple
from tensorneat.common import fetch_first, I_INF
from tensorneat.genome.gene import BaseGene
from tensorneat.utils import extract_gene_attrs, set_gene_attrs


def default_crossover(
    state,
    genome,
    randkey: torch.Generator,
    nodes1: Tensor,
    conns1: Tensor,
    nodes2: Tensor,
    conns2: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Perform crossover between two genomes to generate a new genome.
    Assumes genome1 (nodes1, conns1) has higher fitness than genome2 (nodes2, conns2).
    """
    # Split the random generator for nodes and connections
    randkey1, randkey2 = split_generator(randkey, 2)
    node_randkeys = split_generator(randkey1, genome.max_nodes)
    conn_randkeys = split_generator(randkey2, genome.max_conns)

    # Crossover nodes
    node_keys1 = nodes1[:, : len(genome.node_gene.fixed_attrs)]
    node_keys2 = nodes2[:, : len(genome.node_gene.fixed_attrs)]
    node_attrs1 = torch.stack([extract_gene_attrs(genome.node_gene, node) for node in nodes1])
    node_attrs2 = torch.stack([extract_gene_attrs(genome.node_gene, node) for node in nodes2])

    new_node_attrs = torch.stack([
        create_new_gene(
            state,
            randkey,
            genome.node_gene,
            key1,
            attr1,
            node_keys2,
            node_attrs2,
        )
        for randkey, key1, attr1 in zip(node_randkeys, node_keys1, node_attrs1)
    ])
    new_nodes = torch.stack([
        set_gene_attrs(genome.node_gene, node, new_attr)
        for node, new_attr in zip(nodes1, new_node_attrs)
    ])

    # Crossover connections
    conn_keys1 = conns1[:, : len(genome.conn_gene.fixed_attrs)]
    conn_keys2 = conns2[:, : len(genome.conn_gene.fixed_attrs)]
    conn_attrs1 = torch.stack([extract_gene_attrs(genome.conn_gene, conn) for conn in conns1])
    conn_attrs2 = torch.stack([extract_gene_attrs(genome.conn_gene, conn) for conn in conns2])

    new_conn_attrs = torch.stack([
        create_new_gene(
            state,
            randkey,
            genome.conn_gene,
            key1,
            attr1,
            conn_keys2,
            conn_attrs2,
        )
        for randkey, key1, attr1 in zip(conn_randkeys, conn_keys1, conn_attrs1)
    ])
    new_conns = torch.stack([
        set_gene_attrs(genome.conn_gene, conn, new_attr)
        for conn, new_attr in zip(conns1, new_conn_attrs)
    ])

    return new_nodes, new_conns


def create_new_gene(
    state,
    randkey: torch.Generator,
    gene: BaseGene,
    gene_key: Tensor,
    gene_attrs: Tensor,
    genes_keys: Tensor,
    genes_attrs: Tensor,
) -> Tensor:
    """
    Create a new gene by performing crossover between homologous genes.
    """
    # Find homologous genes
    homologous_idx = fetch_first(torch.all(gene_key == genes_keys, dim=1))

    if homologous_idx == I_INF:  # No homologous gene found, use winner's gene
        return gene_attrs

    # Perform crossover with the homologous gene
    return gene.crossover(state, randkey, gene_attrs, genes_attrs[homologous_idx])