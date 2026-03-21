import random

import networkx as nx
import numpy as np


def generate_complete_dag(
    nodes: int, edge_probability: float = 1, offset: int = 0
) -> nx.DiGraph:
    G = nx.DiGraph()
    perm = np.random.permutation(nodes) + offset
    G.add_nodes_from(perm)

    for i in range(nodes):
        for j in range(i + 1, nodes):
            if random.random() < edge_probability:
                G.add_edge(perm[i], perm[j])

    assert nx.is_directed_acyclic_graph(G)

    return G


def generate_star_dag(nodes: int) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(nodes))
    perm = np.random.permutation(nodes)
    G.add_edges_from((perm[i + 1], perm[0]) for i in range(nodes - 1))

    assert nx.is_directed_acyclic_graph(G)

    return G


def _random_subset(sequence: list[int], amount: int) -> set[int]:
    targets = set()
    while len(targets) < amount:
        x = random.choice(sequence)
        targets.add(x)
    return targets


def hub_node_dag(n: int, m: int) -> nx.DiGraph:
    # adapted from the `barabasi_albert_grap` function in NetowrkX
    assert n >= m

    G = generate_star_dag(m)

    repeated_nodes = [node for node, d in G.degree() for _ in range(d)]
    # Start adding the other n - m0 nodes.
    source = len(G)
    while source < n:
        # Now choose m unique nodes from the existing nodes
        # Pick uniformly from repeated_nodes (preferential attachment)
        targets = _random_subset(repeated_nodes, m)
        # Add edges to m nodes from the source.
        G.add_edges_from(zip([source] * m, targets))
        # Add one node to the list for each new edge just created.
        repeated_nodes.extend(targets)
        # And the new node "source" has m edges to add to the list.
        repeated_nodes.extend([source] * m)

        source += 1

    assert nx.is_directed_acyclic_graph(G)

    return G
