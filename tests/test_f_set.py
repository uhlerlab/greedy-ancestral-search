import logging
from typing import Callable, Optional

import networkx as nx
import pytest

from greedy_ancestral_search.sets import compute_f
from greedy_ancestral_search.utils import break_adjacencies
from tests import generators, utils

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)


def _compute_f(
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
    return compute_f(prefix_set, nodes - prefix_set, ci_tester, n, G, sep_sets)


def test_empty_prefix():
    G = nx.DiGraph()
    nodes = set(range(3))
    G.add_nodes_from(nodes)
    G.add_edges_from([(0, 1), (1, 2)])
    tester = utils.get_ci_tester(G)

    f_sets = set()
    for n in range(1, 30):
        f_set = _compute_f(set(), nodes, tester, n)
        f_sets.update(f_set)

    assert f_sets == set()


@pytest.mark.parametrize("clique_size", range(1, 30))
def test_under_clique(clique_size: int):
    offset = 2
    G = generators.generate_complete_dag(clique_size)
    G.add_nodes_from(range(clique_size, clique_size + offset))

    u = clique_size
    w = clique_size + 1

    for v in range(clique_size):
        G.add_edges_from([(u, v), (v, w)])

    tester = utils.get_ci_tester(G)
    f_set = _compute_f({u}, set(range(clique_size + offset)), tester, clique_size)

    assert f_set == {w}


@pytest.mark.parametrize("children", range(20))
def test_children(children: int):
    offset = 3
    G = generators.generate_complete_dag(children)
    G.add_nodes_from(range(children, children + offset))

    u = children
    v = children + 1
    w = children + 2

    G.add_edges_from([(u, v), (v, w)])
    for d in range(children):
        G.add_edge(w, d)

    tester = utils.get_ci_tester(G)
    f_set = _compute_f({u}, set(range(children + offset)), tester, 1)

    assert f_set == {w} | set(range(children))
