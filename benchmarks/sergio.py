import datetime
import logging

import networkx as nx
import numpy as np
from SERGIO.SERGIO.sergio import sergio

from benchmarks import eval, runners, utils
from benchmarks.utils import Algo

# VARIABLES
RUNS = 5
Y_TITLE = "Average SHD between skeleta"


def run_sergio_simulation():
    """Runs the SERGIO simulation and returns the data and the true graph."""
    sim = sergio(
        number_genes=100,
        number_bins=9,
        number_sc=300,
        noise_params=1,
        decays=0.8,
        sampling_state=15,
        noise_type="dpd",
    )
    sim.build_graph(
        input_file_taregts="benchmarks/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt",
        input_file_regs="benchmarks/SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt",
        shared_coop_state=2,
    )
    sim.simulate()
    expr = sim.getExpressions()
    expr_clean = np.concatenate(expr, axis=1)
    data = expr_clean

    adj_data = {key: value["targets"] for key, value in sim.graph_.items()}

    graph = nx.DiGraph()
    edges_to_add = []
    for source_node, target_nodes in adj_data.items():
        graph.add_node(source_node)
        for target_node in target_nodes:
            edges_to_add.append((source_node, target_node))

    graph.add_edges_from(edges_to_add)

    return data, graph


def main():
    # LOGGING
    filename = f"sergio_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    utils.init_logging(level="INFO", log_filename=filename)
    logger = logging.getLogger(__name__)

    # ALGORITHMS
    algorithm_configs = [
        (
            "causallearn GRASP",
            runners.cl_grasp,
            {},
            eval.shd_cl_gg,
        ),
        (
            "Algorithm with CL tester",
            runners.cl_algorithm,
            {"orientation_method": "ordering", "extra_orientations": False},
            eval.shd,
        ),
        (
            "Algorithm with CL tester with more tests",
            runners.cl_algorithm,
            {"orientation_method": "ci_tests", "extra_orientations": False},
            eval.shd,
        ),
        (
            "Algorithm with CL tester + extra",
            runners.cl_algorithm,
            {"orientation_method": "ordering", "extra_orientations": True},
            eval.shd,
        ),
        (
            "Algorithm with CL tester with more tests + extra",
            runners.cl_algorithm,
            {"orientation_method": "ci_tests", "extra_orientations": True},
            eval.shd,
        ),
    ]

    algorithms = [
        Algo(name, runner, args=args, accuracy_function=acc_func)
        for name, runner, args, acc_func in algorithm_configs
    ]

    # START
    for algo in algorithms:
        algo.times.append([])
        algo.accuracies.append([])

    for j in range(RUNS):
        logger.info(f"Starting run {j + 1} of {RUNS}")
        data, graph = run_sergio_simulation()

        for algo in algorithms:
            run_time, G = algo.caller(data, algo.args)
            algo.times[0].append(run_time / RUNS)
            if algo.accuracy_function is not None:
                algo.accuracies[0].append(algo.accuracy_function(G, graph))

            logger.info(f"{algo.description} time: {algo.times[0][-1]:.4f}")
            logger.info(f"{algo.description} accuracy: {algo.accuracies[0][-1]:.4f}")

    # PLOTS
    utils.plot([100], algorithms, filename, x_axis="Number of Nodes", y_axis=Y_TITLE)


if __name__ == "__main__":
    main()
