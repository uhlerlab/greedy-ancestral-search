import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

_logger = logging.getLogger(__name__)


def init_logging(level: str = "INFO", log_filename: str = "") -> None:
    """Initializes logging for the benchmarks."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_filename:
        os.makedirs("logs/", exist_ok=True)
        handler = logging.FileHandler(f"logs/{log_filename}.log", "w")
    else:
        handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(funcName)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


class SCM:
    """A Structural Causal Model (SCM) for generating data."""

    def __init__(
        self, graph: nx.DiGraph, sigma_square: float, standardize: bool = False
    ):
        assert nx.is_directed_acyclic_graph(graph)

        self._nodes = len(graph)
        self.sigma_square = sigma_square

        if not nx.is_weighted(graph):
            self._add_random_weights(graph)

        self.graph = graph
        self.eps_means = np.zeros(self._nodes)
        self.eps_cov = self.sigma_square * np.identity(self._nodes)
        self.B = np.transpose(
            nx.adjacency_matrix(self.graph, nodelist=list(range(self._nodes)))
        )
        self.A = np.linalg.inv(np.identity(self._nodes) - self.B)
        self.mu = np.matmul(self.A, self.eps_means.reshape(-1, 1))

        if standardize:
            self.standardize_data()

    def _add_random_weights(self, graph: nx.DiGraph) -> None:
        """Adds random weights to the edges of a graph."""
        weights = self._weight_func(len(graph.edges))
        for i, edge in enumerate(graph.edges()):
            graph.add_edge(edge[0], edge[1], weight=weights[i])

    def _weight_func(self, size: int) -> list[float]:
        """Generates random weights for the edges."""
        sgn = np.random.binomial(n=1, p=0.5, size=size)
        low = 0.25
        high = 1
        rand = []
        for i in range(size):
            if sgn[i] == 0:
                rand.append(np.random.uniform(low=-high, high=-low))
            else:
                rand.append(np.random.uniform(low=low, high=high))
        return rand

    def standardize_data(self) -> None:
        """Standardizes the data to have unit variance."""
        scale = (
            np.matmul(np.matmul(self.A, self.eps_cov), self.A.T)
            .diagonal()
            .reshape(-1, 1)
            ** -0.5
        )
        self.eps_means = scale.reshape(-1) * self.eps_means
        self.eps_cov = scale * self.eps_cov * (scale.reshape(-1))
        self.sigma_square = self.eps_cov.diagonal()
        self.B = scale * self.B * ((1 / scale).reshape(-1))
        self.A = scale * self.A * ((1 / scale).reshape(-1))
        self.mu = scale * self.mu

    def sample(self, n: int) -> np.ndarray:
        """Samples data from the SCM."""
        eps = np.random.multivariate_normal(self.eps_means, self.eps_cov, n).reshape(
            self._nodes, n
        )
        batch = np.dot(self.A, eps)
        return batch


@dataclass
class Algo:
    """A data class for storing algorithm information and results."""

    description: str
    caller: Callable
    args: dict[str, Any] = field(default_factory=dict)
    max_nodes: int | None = None
    accuracy_function: Callable | None = None

    def __post_init__(self):
        self.times: list[list[float]] = []
        self.accuracies: list[list[float]] = []
        self.ci_tests: list[list[int]] = []


def _plot_ax(
    ax: plt.Axes,
    x: list[int] | list[float],
    algorithms: list[Algo],
    data_key: str,
    y_label: str,
    x_label: str,
    y_scale: str = "linear",
) -> None:
    """Helper function to plot data on a given axis."""
    for algorithm in algorithms:
        y = getattr(algorithm, data_key)
        y = np.array([np.average(once) for once in y])
        _logger.info(f"{algorithm.description} {data_key}: {y}")
        ax.plot(x[: len(y)], y, label=algorithm.description, marker="o", linewidth=2)

    ax.set_yscale(y_scale)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(title="")
    ax.grid(True)


def plot(
    x: list[int] | list[float],
    algorithms: list[Algo],
    filename: str,
    x_axis: str = "",
    y_axis: str = "",
) -> None:
    """Plots the results of the benchmarks."""
    _, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

    _plot_ax(axs[0], x, algorithms, "times", "Time (Seconds)", x_axis, y_scale="log")
    _plot_ax(axs[1], x, algorithms, "accuracies", y_axis, x_axis)

    os.makedirs("plots/", exist_ok=True)
    plt.savefig(f"plots/{filename}.png")


def save_run(algorithms: list[Algo], title: str, X: list[Any], accuracies: str) -> None:
    """Saves the results of a benchmark run to a file."""
    output_path = os.path.join("out", title)
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "readme"), "w") as f:
        f.write(f"{accuracies=}\n")
        f.write(f"{X=}\n")

    for algorithm in algorithms:
        data = {
            "times": algorithm.times,
            "accuracies": algorithm.accuracies,
        }
        filename = f"{algorithm.description.lower().replace(' ', '-')}.json"
        with open(os.path.join(output_path, filename), "w") as f:
            json.dump(data, f)
