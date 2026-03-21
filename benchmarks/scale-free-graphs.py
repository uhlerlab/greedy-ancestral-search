import datetime
import logging

from benchmarks import eval, runners, utils
from benchmarks.utils import Algo
from tests import generators

# VARIABLES
NODES = 50
EDGES_TO_ADD = [i for i in range(15, 45, 5)]
RUNS = 1
SAMPLES = 10000
Y_TITLE = "Average Structural Hamming Distance"


def main():
    # LOGGING
    filename = (
        f"scale-free-graph_{NODES}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    utils.init_logging(level="INFO", log_filename=filename)
    logger = logging.getLogger(__name__)

    # ALGORITHMS
    algorithm_configs = [
        ("causaldag PC", runners.cd_pc, {}, eval.shd_cd_pdag),
        ("causaldag GSP", runners.cd_gsp, {}, eval.shd_cd_dag),
        (
            "causaldag GSP+",
            runners.cd_gsp,
            {"depth": None, "nruns": 10},
            eval.shd_cd_dag,
        ),
        (
            "Algorithm with CD tester + extra",
            runners.cd_algorithm,
            {"orientation_method": "ordering", "extra_orientations": True},
            eval.shd,
        ),
        (
            "Algorithm with CD tester with more tests + extra",
            runners.cd_algorithm,
            {"orientation_method": "ci_tests", "extra_orientations": True},
            eval.shd,
        ),
        (
            "Algorithm with CD tester",
            runners.cd_algorithm,
            {"orientation_method": "ordering", "extra_orientations": False},
            eval.shd,
        ),
        (
            "Algorithm with CD tester with more tests",
            runners.cd_algorithm,
            {"orientation_method": "ci_tests", "extra_orientations": False},
            eval.shd,
        ),
    ]

    algorithms = [
        Algo(name, runner, args=args, accuracy_function=acc_func)
        for name, runner, args, acc_func in algorithm_configs
    ]

    # START
    for i, _edges_to_add in enumerate(EDGES_TO_ADD):
        logger.info(f"Testing algorithms with {_edges_to_add} edges to add")

        for algo in algorithms:
            algo.times.append([])
            algo.accuracies.append([])

        for j in range(RUNS):
            logger.debug(f"Starting run {j + 1} of {RUNS}")

            # SAMPLES
            graph = generators.hub_node_dag(NODES, _edges_to_add)
            logger.debug(f"{NODES=}, {graph.edges=}")
            model = utils.SCM(graph, 1.0)
            model.standardize_data()
            data = model.sample(SAMPLES)

            for algo in algorithms:
                run_time, G = algo.caller(data, algo.args)
                algo.times[i].append(run_time)
                if algo.accuracy_function is not None:
                    algo.accuracies[i].append(algo.accuracy_function(G, graph))

    logger.info("done")

    utils.save_run(algorithms, filename, EDGES_TO_ADD, accuracies=Y_TITLE)

    # PLOTS
    utils.plot(
        EDGES_TO_ADD,
        algorithms,
        filename,
        x_axis="m parameter",
        y_axis=Y_TITLE,
    )


if __name__ == "__main__":
    main()
