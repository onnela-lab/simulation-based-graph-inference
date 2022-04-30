import argparse
import time
import torch as th
from simulation_based_graph_inference import generators  # Absolute import for line_profiler CLI.


GENERATORS = [
    "generate_duplication_mutation_complementation",
    "generate_duplication_mutation_random",
    "generate_poisson_random_attachment",
    "generate_redirection",
]


def __main__(args: list[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("generator", choices=GENERATORS)
    parser.add_argument("--num_nodes", "-n", help="number of nodes", default=10000, type=int)
    parser.add_argument("--num_samples", "-s", help="number of independent graph samples", type=int)
    args = parser.parse_args(args)

    # Generate parameters for each method.
    generator = getattr(generators, args.generator)
    if generator is generators.generate_duplication_mutation_complementation:
        parameters = {
            "interaction_proba": th.distributions.Beta(1, 9).sample(),
            "divergence_proba": th.distributions.Beta(7, 3).sample(),
        }
    elif generator is generators.generate_duplication_mutation_random:
        parameters = {
            "deletion_proba": th.distributions.Beta(6, 4).sample(),
            "mutation_proba": th.distributions.Beta(2, 8).sample(),
        }
    elif generator is generators.generate_poisson_random_attachment:
        parameters = {
            "rate": th.distributions.Gamma(4, 1).sample(),
        }
    elif generator is generators.generate_redirection:
        parameters = {
            "max_num_connections": 4,
            "redirection_proba": th.distributions.Beta(2, 2).sample()
        }
    else:
        raise NotImplementedError(generator)

    # Set up line profiling if desired.
    try:
        generator = profile(generator)
    except NameError as ex:
        if str(ex) != "name 'profile' is not defined":
            raise  # pragma: no cover

    # Generate samples.
    start = time.time()
    count = 0
    while (args.num_samples and count < args.num_samples) \
            or (args.num_samples is None and time.time() - start < 10):
        generator(args.num_nodes, **parameters)
        count += 1
    duration = time.time() - start
    print(f"generated {count} samples in {duration:.3f} secs ({count / duration:.3f} per sec)")


if __name__ == "__main__":
    __main__()  # pragma: no cover
