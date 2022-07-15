import numpy as np
import pytest
from simulation_based_graph_inference.graph import Graph
from simulation_based_graph_inference import generators
import typing


@pytest.mark.parametrize("num_nodes", [-1, 0, 50, 200])
@pytest.mark.parametrize("generator, args", [
    (generators.generate_poisson_random_attachment, (4,)),
    (generators.generate_duplication_mutation_complementation, (.7, .3)),
    (generators.generate_duplication_mutation_random, (.6, .2)),
    (generators.generate_redirection, (4, .3)),
    (generators.geometric, (lambda dist: dist < .5,))
])
def test_generator(num_nodes: int, generator: typing.Callable, args: list):
    # Check validation against non-positive number of nodes.
    if num_nodes <= 0:
        with pytest.raises(ValueError):
            generator(num_nodes, *args)
    else:
        graph: Graph = generator(num_nodes, *args)
        if isinstance(graph, Graph):
            assert graph.get_num_nodes() == num_nodes
        else:
            assert graph.number_of_nodes() == num_nodes


@pytest.mark.parametrize("generator, args", [
    (generators.generate_poisson_random_attachment, (-1,)),
    (generators.generate_duplication_mutation_complementation, (-.7, .3)),
    (generators.generate_duplication_mutation_complementation, (.7, 1.1)),
    (generators.generate_duplication_mutation_random, (1.6, .2)),
    (generators.generate_duplication_mutation_random, (.6, -.2)),
    (generators.generate_redirection, (0, .3)),
    (generators.generate_redirection, (4, 1.3)),
])
def test_generator_invalid_parameters(generator: typing.Callable, args: list):
    with pytest.raises(ValueError):
        generator(20, *args)


def test_seed():
    edge_sets = []
    for _ in range(2):
        generators.set_seed(42)
        graph = generators.generate_poisson_random_attachment(10, 3)
        edge_sets.append(graph.edges)
    assert edge_sets[0] == edge_sets[1]


@pytest.mark.parametrize('method', [
    generators.adaptive_sample,
    generators.knuth_sample,
    generators.rejection_sample,
])
def test_sample(method):
    population_size = 20
    sample_size = 5
    num_repeats = 10000
    count = 0
    for _ in range(num_repeats):
        sample = list(method(population_size, sample_size))
        count = count + np.bincount(sample, minlength=population_size)
    # Make sure that we get roughly the expected number of hits per bin.
    expected = sample_size * num_repeats / population_size
    np.testing.assert_array_less(.9 * expected, count)
    np.testing.assert_array_less(count, 1.1 * expected)
