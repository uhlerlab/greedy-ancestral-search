# Greedy Ancestral Search

[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-red.svg)]()
[![CI](https://github.com/uhlerlab/greedy-ancestral-search/actions/workflows/ci.yaml/badge.svg)](https://github.com/uhlerlab/greedy-ancestral-search/actions/workflows/ci.yaml)
[![PyPI](https://img.shields.io/pypi/v/greedy-ancestral-search.svg?)](https://pypi.org/project/greedy-ancestral-search/)
[![Python Version](https://img.shields.io/pypi/pyversions/greedy-ancestral-search.svg?)](https://pypi.org/project/greedy-ancestral-search/)

Official implementation of the Greedy Ancestral Seach (GAS) algorithm from the paper **"On the Number of Conditional Independence Tests in Constraint-based Causal Discovery"**.

## Installation

To get started, you'll need Python 3.10 or newer.
You can install `greedy-ancestral-search` directly from PyPI:

```sh
pip install greedy-ancestral-search
```

## Usage

Here is a simple example of using GAS with an oracle d-separation tester to learn the essential graph from a randomly generated directed acyclic graph.

```python
import random
import networkx as nx
from greedy_ancestral_search import greedy_ancestral_search

# 1. Define the set of nodes.
number_of_nodes = 10
nodes = set(range(number_of_nodes))

# 2. Create a random ground-truth DAG to simulate data.
edge_probability = 0.5
G = nx.DiGraph()
G.add_nodes_from(nodes)
for u in nodes:
    for v in nodes:
        if u < v and random.random() < edge_probability:
            G.add_edge(u, v)

# 3. Define a conditional independence tester function.
#    This function queries the ground-truth graph for d-separation.
#    In a real-world scenario, this would be a statistical test on data.
def oracle_tester(X, Y, condition_set):
    """Returns True if X and Y are d-separated by condition_set in G."""
    return nx.is_d_separator(G, X, Y, condition_set)

# 4. Run the GAS algorithm.
#    The function takes the set of nodes and the CI tester as input.
undirected_edges, directed_edges = greedy_ancestral_search(nodes, oracle_tester)

# The algorithm returns the edges of the learned essential graph.
print("Undirected Edges:", undirected_edges)
print("Directed Edges:", directed_edges)
```

## Development

First, clone the repository:

```sh
git clone --recurse-submodules https://github.com/uhlerlab/greedy-ancestral-search.git
cd greedy-ancestral-search
```

### Dependencies

This project uses `uv` for dependency management.
For more information, see the [official documentation](https://docs.astral.sh/uv/).
If you don't have `uv`, install it with the official script:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, install all development and runtime dependencies:
```sh
uv sync --all-groups
```

### Tests

The test suite validates the algorithm's correctness by generating random DAGs and verifying that the algorithm output, using an oracle d-separation tester, corresponds to the correct essential graph.
Tests are located in the `tests/` directory.

To run the standard test suite:
```sh
uv run pytest
```

To run the comprehensive test suite, which uses larger graphs and more iterations (as used in our CI workflow):
```sh
export CI=1 && uv run pytest
```

### Benchmarks

The `benchmarks/` directory contains scripts for performance evaluation.
To run a specific benchmark:
```sh
uv run benchmarks/<script>.py
```

Available scripts include:
- `benchmarks/airfoil.py`
- `benchmarks/neighborhood-comparison.py`
- `benchmarks/nodes-comparison.py`
- `benchmarks/sample-size.py`
- `benchmarks/scale-free-graphs.py`
- `benchmarks/sergio.py`
