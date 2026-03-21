import datetime
import logging

from benchmarks import eval, runners, utils
from benchmarks.utils import Algo
from tests import generators

# VARIABLES
NODES = 50
RUNS = 1
SAMPLES = [100]
EDGE_PROBABILITY = 0.5
Y_TITLE = "SHD"


def main():
    # LOGGING
    filename = (
        f"sample-size_{NODES}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    utils.init_logging(level="INFO", log_filename=filename)
    logger = logging.getLogger(__name__)

    # ALGORITHMS
    algorithm_configs = [
        ("causaldag PC", runners.cd_pc, {}, eval.shd_skel_cd_pdag, None),
        ("causaldag GSP", runners.cd_gsp, {}, eval.shd_skel_cd_dag, None),
        (
            "causaldag GSP+",
            runners.cd_gsp,
            {"depth": None, "nruns": 10},
            eval.shd_skel_cd_dag,
            None,
        ),
        ("causallearn PC", runners.cl_pc, {}, eval.shd_skel_cl_gg, None),
        ("causallearn FCI", runners.cl_fci, {}, eval.shd_skel_cl_gg, None),
        ("causallearn Grasp", runners.cl_grasp, {}, eval.shd_skel_cl_gg, None),
        (
            "Algorithm with CD tester",
            runners.cd_algorithm,
            {"orientation_method": "ordering", "extra_orientations": True},
            eval.shd_skel,
            None,
        ),
        (
            "Algorithm with CD tester with more tests",
            runners.cd_algorithm,
            {"orientation_method": "ci_tests", "extra_orientations": True},
            eval.shd_skel,
            None,
        ),
    ]

    algorithms = [
        Algo(name, runner, args=args, accuracy_function=acc_func, max_nodes=max_nodes)
        for name, runner, args, acc_func, max_nodes in algorithm_configs
    ]

    # START
    for i, samples_ in enumerate(SAMPLES):
        logger.info(f"Testing algorithms with {samples_} samples")

        for algo in algorithms:
            if algo.max_nodes is not None and NODES > algo.max_nodes:
                continue
            algo.times.append([])
            if algo.accuracy_function is not None:
                algo.accuracies.append([])

        for j in range(RUNS):
            logger.info(f"Starting run {j + 1} of {RUNS}")

            # SAMPLES
            graph = generators.generate_complete_dag(NODES, EDGE_PROBABILITY)
            model = utils.SCM(graph, 1.0)
            model.standardize_data()
            data = model.sample(samples_)

            for algo in algorithms:
                if algo.max_nodes is not None and NODES > algo.max_nodes:
                    continue
                run_time, G = algo.caller(data, algo.args)
                algo.times[i].append(run_time / RUNS)
                if algo.accuracy_function is not None:
                    algo.accuracies[i].append(algo.accuracy_function(G, graph))

                logger.info(f"{algo.description} time: {algo.times[i][-1]:.4f}")
                logger.info(
                    f"{algo.description} accuracy: {algo.accuracies[i][-1]:.4f}"
                )

    utils.save_run(algorithms, filename, X=SAMPLES, accuracies=Y_TITLE)

    # PLOTS
    utils.plot(
        SAMPLES, algorithms, filename, x_axis="Number of Samples", y_axis=Y_TITLE
    )


if __name__ == "__main__":
    main()
