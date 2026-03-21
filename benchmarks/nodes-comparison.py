import datetime
import logging

from benchmarks import eval, runners, utils
from benchmarks.utils import Algo
from tests import generators

# VARIABLES
NODES = [5]
RUNS = 1
SAMPLES = 10000
EDGE_PROBABILITY = 0.5
Y_TITLE = "Average SHD"


def main():
    # LOGGING
    filename = f"nodes-comparison_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    utils.init_logging(level="INFO", log_filename=filename)
    logger = logging.getLogger(__name__)

    # ALGORITHMS
    algorithm_configs = [
        (
            "causallearn PC",
            runners.cl_pc,
            {},
            eval.shd_cl_gg,
            200,
        ),
        (
            "causallearn FCI",
            runners.cl_fci,
            {},
            eval.shd_cl_gg,
            200,
        ),
        (
            "causallearn GRASP",
            runners.cl_grasp,
            {},
            eval.shd_cl_gg,
            100,
        ),
        ("causaldag PC", runners.cd_pc, {}, eval.shd_cd_pdag, None),
        (
            "causaldag GSP",
            runners.cd_gsp,
            {},
            eval.shd_cd_dag,
            1000,
        ),
        (
            "causaldag GSP+",
            runners.cd_gsp,
            {"depth": None, "nruns": 10},
            eval.shd_cd_dag,
            1000,
        ),
        (
            "Algorithm with CD tester",
            runners.cd_algorithm,
            {"orientation_method": "ordering", "extra_orientations": True},
            eval.shd,
            None,
        ),
        (
            "Algorithm with CD tester with more tests",
            runners.cd_algorithm,
            {"orientation_method": "ci_tests", "extra_orientations": True},
            eval.shd,
            None,
        ),
        (
            "Algorithm with CL tester",
            runners.cl_algorithm,
            {"orientation_method": "ordering", "extra_orientations": True},
            eval.shd,
            200,
        ),
        (
            "Algorithm with CL tester with more tests",
            runners.cl_algorithm,
            {"orientation_method": "ci_tests", "extra_orientations": True},
            eval.shd,
            200,
        ),
    ]

    algorithms = [
        Algo(name, runner, args=args, accuracy_function=acc_func, max_nodes=max_nodes)
        for name, runner, args, acc_func, max_nodes in algorithm_configs
    ]

    # START
    for i, nodes_ in enumerate(NODES):
        logger.info(f"Testing algorithms with {nodes_} nodes")

        for algo in algorithms:
            if algo.max_nodes is not None and nodes_ > algo.max_nodes:
                continue
            algo.times.append([])
            if algo.accuracy_function is not None:
                algo.accuracies.append([])

        for j in range(RUNS):
            logger.info(f"Starting run {j + 1} of {RUNS}")

            # SAMPLES
            graph = generators.generate_complete_dag(nodes_, EDGE_PROBABILITY)
            model = utils.SCM(graph, 1.0)
            model.standardize_data()
            data = model.sample(SAMPLES)

            for algo in algorithms:
                if algo.max_nodes is not None and nodes_ > algo.max_nodes:
                    continue
                run_time, G = algo.caller(data, algo.args)
                algo.times[i].append(run_time / RUNS)
                if algo.accuracy_function is not None:
                    algo.accuracies[i].append(algo.accuracy_function(G, graph))

                logger.info(f"{algo.description} time: {algo.times[i][-1]:.4f}")
                logger.info(
                    f"{algo.description} accuracy: {algo.accuracies[i][-1]:.4f}"
                )

    utils.save_run(algorithms, filename, X=NODES, accuracies=Y_TITLE)

    # PLOTS
    utils.plot(NODES, algorithms, filename, x_axis="Number of Nodes", y_axis=Y_TITLE)


if __name__ == "__main__":
    main()
