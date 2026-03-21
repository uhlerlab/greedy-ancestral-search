import logging
from typing import Callable, List, Tuple

import graphical_models as gm
import networkx as nx

logger = logging.getLogger(__name__)


def get_ci_tester(G: nx.DiGraph) -> Callable:
    def tester(X, Y, condition_set):
        return nx.is_d_separator(G, X, Y, condition_set)

    return tester


def like_mec(
    undirected_edges: List[Tuple[int, int]],
    directed_edges: List[Tuple[int, int]],
    G: nx.DiGraph,
) -> bool:
    def _sorter(u, v):
        return (u, v) if u < v else (v, u)

    def _converter(iterable):
        return set(_sorter(u, v) for u, v in iterable)

    predicted_edges = _converter(undirected_edges) | _converter(directed_edges)
    graph_edges = _converter(G.edges)

    if predicted_edges != graph_edges:
        return False

    # check that all the v-structures are found
    # a -> b <- c
    for a, b, c in nx.dag.v_structures(G):
        if (a, b) not in directed_edges or (c, b) not in directed_edges:
            return False

    return True


def like_essential_graph(
    undirected_edges: List[Tuple[int, int]],
    directed_edges: List[Tuple[int, int]],
    G: nx.DiGraph,
) -> bool:
    essential_graph = gm.DAG(nodes=set(G.nodes), arcs=G.edges).cpdag()

    eg_undirected_edges = essential_graph.edges
    eg_directed_edges = essential_graph.arcs
    formatted_undirected_edges = set(frozenset({u, v}) for u, v in undirected_edges)
    formatted_directed_edges = set((u, v) for u, v in directed_edges)

    logger.debug(f"EG undirected edges:\n{eg_undirected_edges}")
    logger.debug(f"EG directed edges:\n{eg_directed_edges}")
    logger.debug(f"Found undirected edges:\n{formatted_undirected_edges}")
    logger.debug(f"Found directed edges:\n{formatted_directed_edges}")

    return (
        formatted_undirected_edges == eg_undirected_edges
        and formatted_directed_edges == eg_directed_edges
    )
