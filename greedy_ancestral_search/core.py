import itertools
import logging
from typing import Any, Callable

import networkx as nx

from . import sets, utils

logger = logging.getLogger(__name__)


def greedy_ancestral_search(
    nodes: set[int],
    ci_tester: Callable[[int | set[int], int | set[int], int | set[int]], bool],
    orientation_method: str = "ordering",
    extra_orientations: bool = False,
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    """Performs the Greedy Ancestral Search (GAS) algorithm to discover a causal graph.

    Args:
        nodes: A set of variables/nodes.
        ci_tester: A function `ci_tester(X, Y, Z)` that returns True if X and Y
                   are conditionally independent given the set Z, and False otherwise.
        orientation_method: The method to use to orient the edges.
                            - "ordering": Orients edges based on the computed ordering S.
                              Adjacencies are taken from the skeleton graph `G` which has
                              been pruned during the search.
                            - "ci_tests": Uses additional conditional independence tests
                              for orientation.
        extra_orientations: If True, applies additional orientation rules for v-structures
                            to orient more edges.

    Returns:
        A tuple containing two lists:
        - undirected_edges: List of tuples (u, v) representing undirected edges.
        - directed_edges: List of tuples (u, v) representing directed edges (u -> v).
    """
    prefix_set = set()
    S = []

    G = nx.complete_graph(nodes)
    sep_sets = {}

    # construct the prefix set
    while prefix_set != nodes:
        logger.debug(
            f"Finding S_{len(S) + 1}, {len(nodes) - len(prefix_set)} nodes remain"
        )

        current_nodes = nodes - prefix_set
        downstream_nodes = set()

        n = 0
        while utils.exists_k_clique(G, current_nodes, n):
            logger.debug(f"Has a clique of size {n}")

            utils.break_adjacencies(
                prefix_set, current_nodes, ci_tester, n, G, sep_sets
            )

            downstream_nodes_d = sets.compute_d(
                prefix_set, current_nodes, ci_tester, n, G, sep_sets
            )

            downstream_nodes_f = set()
            if n > 0:
                downstream_nodes_f = sets.compute_f(
                    prefix_set,
                    current_nodes,
                    ci_tester,
                    n,
                    G,
                    sep_sets,
                )

            new_downstream_nodes = downstream_nodes_d | downstream_nodes_f
            # should not happen if faithfulness is assumed
            # but in practice can happen and enters an infinite loop
            if current_nodes == new_downstream_nodes:
                valid_d = (
                    not current_nodes == downstream_nodes_d
                    and len(downstream_nodes_d) > 0
                )
                valid_f = (
                    not current_nodes == downstream_nodes_f
                    and len(downstream_nodes_f) > 0
                )
                if valid_d:
                    new_downstream_nodes = downstream_nodes_d
                elif valid_f:
                    new_downstream_nodes = downstream_nodes_f
                else:
                    logger.debug("Breaking out of loop")
                    break

            downstream_nodes |= new_downstream_nodes
            current_nodes -= new_downstream_nodes
            n += 1

        logger.debug(
            f"S_{len(S) + 1} has {len(nodes - (prefix_set | downstream_nodes))} nodes"
        )
        logger.debug(f"S_{len(S) + 1}: {nodes - (prefix_set | downstream_nodes)}")
        S.append(nodes - (prefix_set | downstream_nodes))
        assert len(S[-1]) > 0

        prefix_set.update(nodes - downstream_nodes)

    logger.debug(f"{S=}")

    return _get_orientations(
        ci_tester, G, S, sep_sets, orientation_method, extra_orientations
    )


def _get_orientations(
    ci_tester: Callable,
    G: nx.Graph,
    S: list[set[int]],
    sep_sets: dict[frozenset[int], set[int]],
    orientation_method: str,
    extra_orientations: bool,
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    if orientation_method == "ordering":
        edges = _orientation_by_ordering(G, S)
    elif orientation_method == "ci_tests":
        edges = _orientation_by_ci_tests(ci_tester, S)
    else:
        raise ValueError(f"Invalid orientation method: {orientation_method}")

    if extra_orientations:
        edges = _orient_extra(*edges, ci_tester, G, S, sep_sets)

    return edges


def _orientation_by_ordering(
    G: nx.Graph, S: list[set[int]]
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    undirected_edges = []
    for expansion in S:
        undirected_edges += [
            (u, v) for u, v in itertools.combinations(expansion, 2) if G.has_edge(u, v)
        ]
    directed_edges = []
    for i, j in itertools.combinations(range(len(S)), 2):
        directed_edges += [
            (u, v) for u, v in itertools.product(S[i], S[j]) if G.has_edge(u, v)
        ]
    return undirected_edges, directed_edges


def _orientation_by_ci_tests(
    ci_tester: Callable, S: list[set[int]]
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    undirected_edges = []
    for i, expansion in enumerate(S):
        undirected_edges += [
            (u, v)
            for u, v in itertools.combinations(expansion, 2)
            if not ci_tester(u, v, set().union(*S[: i + 1]) - {u, v})
        ]
    directed_edges = []
    for i, j in itertools.combinations(range(len(S)), 2):
        directed_edges += [
            (u, v)
            for u, v in itertools.product(S[i], S[j])
            if not ci_tester(u, v, set().union(*S[: j + 1]) - {u, v})
        ]
    return undirected_edges, directed_edges


def _orient_extra(
    undirected_edges: list[tuple[Any, Any]],
    directed_edges: list[tuple[Any, Any]],
    ci_tester: Callable,
    G: nx.Graph,
    S: list[set[int]],
    sep_sets: dict[frozenset[int], set[int]],
) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
    # convert to sets for O(1) operations
    undirected_set = set(undirected_edges)
    directed_set = set(directed_edges)

    G = nx.Graph(undirected_edges)
    G.add_nodes_from(set().union(*S))

    # cache neighbor sets to avoid repeated conversions
    neighbor_cache = {}

    for i, expansion in enumerate(S):
        for u, v in itertools.combinations(expansion, 2):
            if G.has_edge(u, v):
                continue

            # use cached neighbor sets
            if u not in neighbor_cache:
                neighbor_cache[u] = set(G.neighbors(u))
            if v not in neighbor_cache:
                neighbor_cache[v] = set(G.neighbors(v))

            common_neighbors = expansion & neighbor_cache[u] & neighbor_cache[v]

            for k in common_neighbors:
                sep_key = frozenset((u, v))
                if sep_key in sep_sets and k in sep_sets[sep_key]:
                    continue

                # check and add directed edges using sets for O(1) operations
                if (u, k) not in directed_set and (k, u) not in directed_set:
                    directed_set.add((u, k))
                if (v, k) not in directed_set and (k, v) not in directed_set:
                    directed_set.add((v, k))

                # remove edges from undirected set
                edges_to_remove = [(u, k), (k, u), (v, k), (k, v)]
                for edge in edges_to_remove:
                    undirected_set.discard(edge)

    # convert back to lists
    return list(undirected_set), list(directed_set)
