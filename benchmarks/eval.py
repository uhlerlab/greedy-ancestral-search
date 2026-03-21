import itertools
import logging

import graphical_models as gm
import networkx as nx
from causallearn.graph.Graph import Graph

logger = logging.getLogger(__name__)


def _get_essential_graph(graph: nx.DiGraph) -> gm.PDAG:
    """Converts a networkx DiGraph to a graphical_models PDAG (essential graph)."""
    return gm.DAG(nodes=set(graph.nodes), arcs=graph.edges).cpdag()


def _calculate_nx_shd(graph1: nx.Graph, graph2: nx.Graph) -> int:
    """Calculates the Structural Hamming Distance between two networkx Graphs."""

    def _sorter(u, v):
        return (u, v) if u < v else (v, u)

    edges1 = {_sorter(u, v) for u, v in graph1.edges()}
    edges2 = {_sorter(u, v) for u, v in graph2.edges()}

    return len(edges1.symmetric_difference(edges2))


def like_essential_graph_cd(pdag: gm.PDAG, underlying_graph: nx.DiGraph) -> bool:
    """Checks if a graphical_models PDAG is the same as the essential graph of a networkx DiGraph."""
    essential_graph = _get_essential_graph(underlying_graph)
    return pdag == essential_graph


def shd_cd_pdag(pdag: gm.PDAG, underlying_graph: nx.DiGraph) -> int:
    """Calculates the Structural Hamming Distance between a graphical_models PDAG and the essential graph of a networkx DiGraph."""
    essential_graph = _get_essential_graph(underlying_graph)
    return essential_graph.shd(pdag)


def shd_skel_cd_pdag(pdag: gm.PDAG, underlying_graph: nx.DiGraph) -> int:
    """Calculates the Structural Hamming Distance between the skeleton of a graphical_models PDAG and a networkx DiGraph."""
    pdag_skeleton = nx.Graph()
    pdag_skeleton.add_edges_from(pdag.arcs | pdag.edges)
    return _calculate_nx_shd(pdag_skeleton, underlying_graph)


def shd_cd_dag(dag: gm.DAG, underlying_graph: nx.DiGraph) -> int:
    """Calculates the Structural Hamming Distance between a graphical_models DAG and a networkx DiGraph."""
    essential_graph = _get_essential_graph(underlying_graph)
    return essential_graph.shd(dag.cpdag())


def shd_skel_cd_dag(dag: gm.DAG, underlying_graph: nx.DiGraph) -> int:
    """Calculates the Structural Hamming Distance between the skeleton of a graphical_models DAG and a networkx DiGraph."""
    dag_skeleton = nx.Graph()
    dag_skeleton.add_edges_from(dag.arcs)
    return _calculate_nx_shd(dag_skeleton, underlying_graph)


def shd_cl_gg(causal_learn_graph: Graph, underlying_graph: nx.DiGraph) -> int:
    """Calculates the Structural Hamming Distance between a causallearn Graph and the essential graph of a networkx DiGraph."""
    essential_graph = _get_essential_graph(underlying_graph)
    adj_mat = causal_learn_graph.graph
    directed_edges = []
    undirected_edges = []
    for i, j in itertools.combinations(range(len(adj_mat)), 2):
        if adj_mat[i, j] == -1 and adj_mat[j, i] == 1:
            directed_edges.append((i, j))
        elif adj_mat[i, j] == 1 and adj_mat[j, i] == -1:
            directed_edges.append((j, i))
        elif adj_mat[i, j] != 0 and adj_mat[j, i] != 0:
            undirected_edges.append((i, j))
    pdag = gm.PDAG(arcs=directed_edges, edges=undirected_edges)

    return essential_graph.shd(pdag)


def shd_skel_cl_gg(causal_learn_graph: Graph, underlying_graph: nx.DiGraph) -> int:
    """Calculates the Structural Hamming Distance between the skeleton of a causallearn Graph and a networkx DiGraph."""
    graph_skeleton = nx.Graph()
    adj_mat = causal_learn_graph.graph
    for i in range(len(adj_mat)):
        for j in range(i + 1, len(adj_mat)):
            if adj_mat[i, j] == 1 or adj_mat[i, j] == -1 or adj_mat[i, j] == 2:
                graph_skeleton.add_edge(i, j)

    return _calculate_nx_shd(graph_skeleton, underlying_graph)


def shd(
    graph: tuple[list[tuple[int, int]], list[tuple[int, int]]],
    underlying_graph: nx.DiGraph,
) -> int:
    """Calculates the Structural Hamming Distance between a tuple of edge lists and the essential graph of a networkx DiGraph."""
    essential_graph = _get_essential_graph(underlying_graph)
    undirected_edges, directed_edges = graph
    pdag = gm.PDAG(arcs=directed_edges, edges=undirected_edges)
    return essential_graph.shd(pdag)


def shd_skel(
    graph: tuple[list[tuple[int, int]], list[tuple[int, int]]],
    underlying_graph: nx.DiGraph,
) -> int:
    """Calculates the Structural Hamming Distance between the skeleton of a tuple of edge lists and a networkx DiGraph."""
    undirected_edges, directed_edges = graph
    graph_skeleton = nx.Graph()
    graph_skeleton.add_edges_from(undirected_edges + directed_edges)
    return _calculate_nx_shd(graph_skeleton, underlying_graph)


def f1_score(
    graph: gm.PDAG | tuple[list[tuple[int, int]], list[tuple[int, int]]],
    underlying_graph: nx.DiGraph,
) -> float:
    """Calculates the F1 score of the skeleton of a graph."""
    if isinstance(graph, tuple):
        undirected_edges, directed_edges = graph
        pdag = gm.PDAG(arcs=directed_edges, edges=undirected_edges)
    else:
        pdag = graph

    confusion_matrix = gm.DAG(
        nodes=set(underlying_graph.nodes), arcs=underlying_graph.edges
    ).confusion_matrix_skeleton(pdag)

    precision = confusion_matrix["num_true_positives"] / (
        confusion_matrix["num_true_positives"] + confusion_matrix["num_false_positives"]
    )
    recall = confusion_matrix["num_true_positives"] / (
        confusion_matrix["num_true_positives"] + confusion_matrix["num_false_negatives"]
    )
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    logger.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    return f1
