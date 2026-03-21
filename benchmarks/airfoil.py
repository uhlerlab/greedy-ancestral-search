import datetime
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from benchmarks import runners, utils

# LOGGING
FILENAME = "real-world_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
utils.init_logging(level="INFO", log_filename=FILENAME)
LOGGER = logging.getLogger(__name__)

ALPHA = 0.0001


def save_graph(file, D, U, names):
    pos = nx.circular_layout(names)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    nx.draw(
        D,
        arrowsize=12,
        with_labels=True,
        node_size=500,
        node_color="#ffffff",
        linewidths=2.0,
        width=1.5,
        font_size=10,
        pos=pos,
        ax=ax,
    )
    nx.draw(
        U,
        pos=pos,
        linewidths=2.0,
        width=1.5,
        font_size=10,
        node_color="#ffffff",
        with_labels=True,
        ax=ax,
    )

    fig.savefig(file)
    plt.close(fig)


def adj_mat_to_graph(adj_mat, names):
    D = nx.DiGraph()
    U = nx.Graph()
    for i in range(len(adj_mat)):
        for j in range(i + 1, len(adj_mat)):
            # https://github.com/py-why/causal-learn/blob/474437c427c5823610c6ca6a03a7fe2a0420db1a/causallearn/graph/GeneralGraph.py#L683
            if adj_mat[i, j] == -1 and adj_mat[j, i] == -1:
                U.add_edge(names[i], names[j])
            elif adj_mat[j, i] == 1 and adj_mat[i, j] == -1:
                D.add_edge(names[i], names[j])
            elif adj_mat[i, j] == 1 and adj_mat[j, i] == -1:
                D.add_edge(names[j], names[i])
    return D, U


def run_and_save(
    runner,
    data,
    names,
    output_filename,
    runner_args=None,
    is_cl_algorithm=False,
):
    if runner_args is None:
        runner_args = {}
    _, graph = runner(data, **runner_args)

    if is_cl_algorithm:
        undir, dir = graph
        named_dir = [(names[i], names[j]) for (i, j) in dir]
        named_undir = [(names[i], names[j]) for (i, j) in undir]
        D = nx.DiGraph(named_dir)
        U = nx.Graph(named_undir)
    else:
        adj_mat = graph.graph
        D, U = adj_mat_to_graph(adj_mat, names)

    save_graph(output_filename, D, U, names)
    LOGGER.info(f"Saved graph to {output_filename}")


def main():
    file = "https://raw.githubusercontent.com/cmu-phil/example-causal-datasets/refs/heads/main/real/airfoil-self-noise/data/airfoil-self-noise.continuous.txt"
    samples = np.loadtxt(file, skiprows=1)
    names = np.loadtxt(file, max_rows=1, dtype=str)

    LOGGER.info(
        f"Loaded data with {samples.shape[0]} samples and {samples.shape[1]} variables."
    )

    run_and_save(
        runners.cl_algorithm,
        samples.T,
        names,
        "real-world.png",
        runner_args={"alpha": ALPHA, "args": {"orientation_method": "ordering"}},
        is_cl_algorithm=True,
    )

    run_and_save(
        runners.cl_algorithm,
        samples.T,
        names,
        "final-tests-real-world.png",
        runner_args={"alpha": ALPHA, "args": {"orientation_method": "ci_tests"}},
        is_cl_algorithm=True,
    )

    run_and_save(runners.cl_grasp, samples.T, names, "grasp-real-world.png")

    run_and_save(
        runners.cl_pc,
        samples.T,
        names,
        "pc-real-world.png",
        runner_args={"args": {"alpha": ALPHA}},
    )


if __name__ == "__main__":
    main()
