import time
from typing import Any, Callable, Tuple

import numpy as np
from causallearn.graph.Graph import Graph as CausalLearnGraph
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.PermutationBased.GRaSP import grasp
from causallearn.utils.cit import CIT
from conditional_independence import (
    MemoizedCI_Tester,
    partial_correlation_suffstat,
    partial_correlation_test,
)
from graphical_model_learning import gsp, pcalg, sparsest_permutation
from graphical_model_learning.algorithms.dag.ges import PDAG

from greedy_ancestral_search import greedy_ancestral_search


def _create_partial_correlation_tester(
    data: np.ndarray, alpha: float = 0.005
) -> MemoizedCI_Tester:
    """Creates a MemoizedCI_Tester for partial correlation tests."""
    suffstat = partial_correlation_suffstat(data.T)
    return MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=alpha)


def _run_algorithm(
    algorithm: Callable[..., Any],
    data: np.ndarray,
    ci_tester_creator: Callable[[np.ndarray], Any],
    args: dict[str, Any] = {},
) -> Tuple[float, Any]:
    """Generic runner for causal discovery algorithms."""
    ci_tester = ci_tester_creator(data)
    start_time = time.time()
    graph = algorithm(set(range(len(data))), ci_tester, **args)
    run_time = time.time() - start_time
    return run_time, graph


def cd_pc(data: np.ndarray, args: dict[str, Any] = {}) -> Tuple[float, PDAG]:
    """Runs the PC algorithm from the causaldag library."""
    return _run_algorithm(pcalg, data, _create_partial_correlation_tester, args)


def cd_gsp(data: np.ndarray, args: dict[str, Any] = {}) -> Tuple[float, PDAG]:
    """Runs the GSP algorithm from the causaldag library."""
    return _run_algorithm(gsp, data, _create_partial_correlation_tester, args)


def cd_sp(data: np.ndarray, args: dict[str, Any] = {}) -> Tuple[float, Any]:
    """Runs the sparsest_permutation algorithm from the causaldag library."""
    return _run_algorithm(
        sparsest_permutation, data, _create_partial_correlation_tester, args
    )


def cd_algorithm(data: np.ndarray, args: dict[str, Any] = {}) -> Tuple[float, Any]:
    """Runs the greedy ancestral search algorithm with a causaldag CI tester."""

    def _ci_tester_wrapper(
        ci_tester: MemoizedCI_Tester,
    ) -> Callable[[Any, Any, Any], bool]:
        def _ci_tester(X, Y, S) -> bool:
            return ci_tester.is_ci(X, Y, S)

        return _ci_tester

    ci_tester = _create_partial_correlation_tester(data)
    wrapped_tester = _ci_tester_wrapper(ci_tester)

    start_time = time.time()
    graph = greedy_ancestral_search(set(range(len(data))), wrapped_tester, **args)
    run_time = time.time() - start_time
    return run_time, graph


def cl_pc(
    data: np.ndarray, args: dict[str, Any] = {}
) -> Tuple[float, CausalLearnGraph]:
    """Runs the PC algorithm from the causallearn library."""
    start_time = time.time()
    graph = pc(data.T, **args)
    run_time = time.time() - start_time
    return run_time, graph.G


def cl_fci(
    data: np.ndarray, args: dict[str, Any] = {}
) -> Tuple[float, CausalLearnGraph]:
    """Runs the FCI algorithm from the causallearn library."""
    start_time = time.time()
    graph, _ = fci(data.T, **args)
    run_time = time.time() - start_time
    return run_time, graph


def cl_grasp(
    data: np.ndarray, args: dict[str, Any] = {}
) -> Tuple[float, CausalLearnGraph]:
    """Runs the GRaSP algorithm from the causallearn library."""
    start_time = time.time()
    graph = grasp(data.T, **args)
    run_time = time.time() - start_time
    return run_time, graph


def cl_algorithm(
    data: np.ndarray, args: dict[str, Any] = {}, alpha: float = 0.05
) -> Tuple[float, Any]:
    """Runs the greedy ancestral search algorithm with a causallearn CI tester."""
    start_time = time.time()
    ci_tester = CIT(data.T)

    def _ci_tester(X, Y, Z):
        return ci_tester(X, Y, Z) > alpha

    graph = greedy_ancestral_search(set(range(len(data))), _ci_tester, **args)
    run_time = time.time() - start_time
    return run_time, graph
