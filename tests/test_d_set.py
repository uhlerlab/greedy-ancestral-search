import logging
import random
from typing import Callable, Optional

import networkx as nx
import pytest

from greedy_ancestral_search.sets import compute_d
from greedy_ancestral_search.utils import break_adjacencies
from tests import generators, utils

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)


def _compute_d(
    prefix_set: set[int],
    nodes: set[int],
    ci_tester: Callable,
    n,
    G: Optional[nx.Graph] = None,
    sep_sets: Optional[dict[frozenset[int], set[int]]] = None,
) -> set[int]:
    if G is None:
        G = nx.complete_graph(prefix_set | nodes)
    if sep_sets is None:
        sep_sets = {}

    break_adjacencies(prefix_set, nodes - prefix_set, ci_tester, n, G, sep_sets)
    return compute_d(prefix_set, nodes - prefix_set, ci_tester, n, G, sep_sets)


def test_u_in_prefix():
    G = nx.DiGraph()
    nodes = set(range(3))
    G.add_nodes_from(nodes)

    v_structure = 0
    G.add_edges_from([(v_structure + 1, v_structure), (v_structure + 2, v_structure)])
    tester = utils.get_ci_tester(G)

    prefix_set = {v_structure + 1}

    d_set = _compute_d(prefix_set, nodes - prefix_set, tester, 0)

    assert d_set == {v_structure}


@pytest.mark.parametrize("clique_size", range(15))
def test_under_clique(clique_size: int):
    offset = 3
    G = generators.generate_complete_dag(clique_size)
    G.add_nodes_from(range(clique_size, clique_size + offset))

    v_structure = clique_size

    down_clique_edges = [
        (i, v_structure + j) for i in range(clique_size) for j in (1, 2)
    ]
    G.add_edges_from(
        [
            (v_structure + 1, v_structure),
            (v_structure + 2, v_structure),
            *down_clique_edges,
        ]
    )

    prefix_set = set()
    nodes = set(range(clique_size + offset))
    tester = utils.get_ci_tester(G)
    d_set = _compute_d(prefix_set, nodes - prefix_set, tester, clique_size)

    assert d_set == {v_structure}


@pytest.mark.parametrize(
    "extra_nodes,edge_probability", [(25, random.random()) for _ in range(5)]
)
def test_descendants(extra_nodes: int, edge_probability: float):
    offset = 3
    G = generators.generate_complete_dag(extra_nodes)

    v_structure = extra_nodes
    edges_to_descendants = [
        (v_structure, i)
        for i in range(extra_nodes)
        if random.random() < edge_probability
    ]
    G.add_edges_from(
        [
            (v_structure + 1, v_structure),
            (v_structure + 2, v_structure),
            *edges_to_descendants,
        ]
    )

    prefix_set = set()
    nodes = set(range(offset + extra_nodes))
    tester = utils.get_ci_tester(G)
    d_set = _compute_d(prefix_set, nodes, tester, 0)

    assert d_set == {v_structure} | nx.descendants(G, v_structure)


@pytest.mark.parametrize(
    "nodes,edge_probability", [(10, random.random()) for _ in range(5)]
)
def test_generic(nodes: int, edge_probability: float):
    G = generators.generate_complete_dag(nodes, edge_probability)
    tester = utils.get_ci_tester(G)

    expected = set()
    for v_structure in nx.dag.v_structures(G):
        node = v_structure[1]
        descendants = nx.descendants(G, node)
        expected.update([node] + list(descendants))

    prefix_set = set()
    nodes_set = set(range(nodes))

    sep_sets = {}
    Gp = nx.complete_graph(nodes)
    d_sets = set()
    n = 0

    while n < nodes - 2 and d_sets != expected:
        d_sets |= _compute_d(prefix_set, nodes_set, tester, n, Gp, sep_sets)
        n += 1

    assert d_sets == expected
