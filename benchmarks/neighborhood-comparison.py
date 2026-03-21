import datetime
import logging

from benchmarks import eval, runners, utils
from benchmarks.utils import Algo
from tests import generators

# VARIABLES
NODES = 50
NEIGHBORHOOD_SIZES = [5]
RUNS = 1
SAMPLES = 10000
Y_TITLE = "Average Structural Hamming Distance"


def main():
    # LOGGING
    filename = f"neighborhood-comparison_{NODES}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
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
        ("causallearn PC", runners.cl_pc, {}, eval.shd_cl_gg),
        ("causallearn FCI", runners.cl_fci, {}, eval.shd_cl_gg),
        ("causallearn Grasp", runners.cl_grasp, {}, eval.shd_cl_gg),
        (
            "Algorithm with CD tester",
            runners.cd_algorithm,
            {"orientation_method": "ordering", "extra_orientations": True},
            eval.shd,
        ),
        (
            "Algorithm with CD tester with more tests",
            runners.cd_algorithm,
            {"orientation_method": "ci_tests", "extra_orientations": True},
            eval.shd,
        ),
        (
            "Algorithm with CL tester",
            runners.cl_algorithm,
            {"orientation_method": "ordering", "extra_orientations": True},
            eval.shd,
        ),
        (
            "Algorithm with CL tester with more tests",
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
    for i, neighborhood_size in enumerate(NEIGHBORHOOD_SIZES):
        logger.info(
            f"Testing algorithms with average neighborhood size {neighborhood_size}"
        )

        for algo in algorithms:
            algo.times.append([])
            algo.accuracies.append([])
            algo.ci_tests.append([])

        edge_probability = neighborhood_size / (NODES - 1)
        for j in range(RUNS):
            logger.debug(f"Starting run {j + 1} of {RUNS}")

            # SAMPLES
            graph = generators.generate_complete_dag(NODES, edge_probability)
            logger.debug(f"{NODES=}, {graph.edges=}")
            model = utils.SCM(graph, 1.0)
            model.standardize_data()

            for algo in algorithms:
                data = model.sample(SAMPLES)
                run_time, G = algo.caller(data, algo.args)
                algo.times[i].append(run_time)
                if algo.accuracy_function is not None:
                    algo.accuracies[i].append(algo.accuracy_function(G, graph))

    utils.save_run(algorithms, filename, NEIGHBORHOOD_SIZES, accuracies=Y_TITLE)

    # PLOTS
    utils.plot(
        NEIGHBORHOOD_SIZES,
        algorithms,
        filename,
        x_axis="Expected neighborhood size",
        y_axis=Y_TITLE,
    )


if __name__ == "__main__":
    main()
