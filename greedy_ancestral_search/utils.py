import itertools
import logging
from typing import Callable

import networkx as nx

logger = logging.getLogger(__name__)


def exists_k_clique(G: nx.Graph, set_: set[int], k: int) -> bool:
    # adapted from the `find_cliques` function in NetworkX
    if k == 0:
        return True
    if not set_:
        return False

    adjacencies = {u: {v for v in G[u] if v in set_} for u in set_}

    candidates = set_.copy()
    subg = candidates.copy()
    u = max(subg, key=lambda u: len(candidates & adjacencies[u]))
    non_adjacent_u = candidates - adjacencies[u]

    stack = []
    Q = []
    Q.append(None)

    try:
        while True:
            if non_adjacent_u:
                q = non_adjacent_u.pop()
                candidates.remove(q)
                Q[-1] = q
                if len(Q) == k:
                    return True

                adjacencies_q = adjacencies[q]
                subg_q = subg & adjacencies_q
                if not subg_q:
                    continue
                else:
                    candidates_q = candidates & adjacencies_q
                    if candidates_q:
                        stack.append((subg, candidates, non_adjacent_u))
                        Q.append(None)
                        subg = subg_q
                        candidates = candidates_q
                        u = max(subg, key=lambda u: len(candidates & adjacencies[u]))
                        non_adjacent_u = candidates - adjacencies[u]
            else:
                Q.pop()
                subg, candidates, non_adjacent_u = stack.pop()
    except IndexError:
        pass

    return False


def connected_nodes(G: nx.Graph, source: set[int], valid_nodes: set[int]) -> set[int]:
    visited = source.copy()
    stack = [
        neighbor
        for s in source
        for neighbor in G[s]
        if neighbor in valid_nodes and neighbor not in visited
    ]

    while stack:
        node = stack.pop()

        visited.add(node)
        neighbors = [
            neighbor
            for neighbor in G[node]
            if neighbor in valid_nodes and neighbor not in visited
        ]
        stack.extend(neighbors)

    return visited - source


def break_adjacencies(
    prefix_set: set[int],
    downstream_set: set[int],
    ci_tester: Callable,
    cond_set_size: int,
    G: nx.Graph,
    sep_sets: dict[frozenset[int], set[int]],
) -> None:
    assert prefix_set & downstream_set == set()

    pairs = itertools.chain(
        itertools.product(prefix_set, downstream_set),
        itertools.combinations(downstream_set, 2),
    )

    for u, v in pairs:
        if not G.has_edge(u, v):
            continue

        for cond_set in itertools.combinations(downstream_set - {u, v}, cond_set_size):
            cond_set_ = set(cond_set)
            if not ci_tester(u, v, prefix_set - {u} | cond_set_):
                continue

            logger.debug(f"Removing edge between {u} and {v}, CI given {cond_set_}")
            G.remove_edge(u, v)
            sep_sets[frozenset((u, v))] = cond_set_
            break
