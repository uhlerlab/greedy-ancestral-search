import logging
import os
import random
from typing import Any

import networkx as nx
import pytest

from greedy_ancestral_search import greedy_ancestral_search
from tests import generators, graphs, utils

MIN_NODES = 40 if "CI" in os.environ else 10
MAX_NODES = 50 if "CI" in os.environ else 20
ITERATIONS = 15 if "CI" in os.environ else 5

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def test_input_format():
    nx_graph = generators.generate_complete_dag(1)
    tester = utils.get_ci_tester(nx_graph)
    greedy_ancestral_search({0}, tester)

    assert True


@pytest.mark.parametrize(
    "nodes,edge_probability",
    [(5, 0.3)],
)
def test_output_format(nodes: int, edge_probability: float):
    G = generators.generate_complete_dag(nodes, edge_probability)
    tester = utils.get_ci_tester(G)
    undirected_edges, directed_edges = greedy_ancestral_search(
        set(range(nodes)), tester
    )

    def format_(edges):
        return isinstance(edges, list) and all(
            isinstance(e, tuple)
            and len(e) == 2
            and isinstance(e[0], int)
            and isinstance(e[1], int)
            for e in edges
        )

    assert format_(undirected_edges) and format_(directed_edges)


RANDOM_GRAPHS = [
    *[
        generators.generate_complete_dag(
            random.randint(MIN_NODES, MAX_NODES), random.random() / 2
        )
        for _ in range(ITERATIONS)
    ],
    *[
        generators.hub_node_dag(
            (n := random.randint(MIN_NODES, MAX_NODES) // 2), n // 2
        )
        for _ in range(ITERATIONS)
    ],
]

GRAPHS = [*RANDOM_GRAPHS, *graphs.GRAPHS]


@pytest.mark.parametrize(
    "G,params",
    [
        (G, {"orientation_method": method, "extra_orientations": extra_orientations})
        for G in GRAPHS
        for method in ("ordering", "ci_tests")
        for extra_orientations in (True, False)
    ],
)
def test_output(G: nx.DiGraph, params: dict[str, Any]):
    logger.debug(G.edges)
    tester = utils.get_ci_tester(G)
    undirected_edges, directed_edges = greedy_ancestral_search(
        set(G.nodes), tester, **params
    )

    assert utils.like_mec(undirected_edges, directed_edges, G)
    assert utils.like_essential_graph(undirected_edges, directed_edges, G)
