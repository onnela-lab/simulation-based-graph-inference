import time

# Absolute import for line_profiler CLI.
from simulation_based_graph_inference import config
from simulation_based_graph_inference.scripts import util


def __main__(argv: list[str] | None = None):
    parser = util.get_parser()
    parser.add_argument(
        "--num_nodes", "-n", help="number of nodes", type=int, required=True
    )
    parser.add_argument(
        "--num_samples", "-m", help="number of independent graph samples", type=int
    )
    args = parser.parse_args(argv)

    # Generate parameters for each method.
    generator_configuration = config.GENERATOR_CONFIGURATIONS[args.configuration]

    # Set up line profiling if desired.
    try:
        generator_configuration.generator = profile(generator_configuration.generator)  # pyright: ignore[reportUndefinedVariable]
    except NameError as ex:
        if str(ex) != "name 'profile' is not defined":
            raise  # pragma: no cover

    # Generate samples.
    start = time.time()
    count = 0
    while (args.num_samples and count < args.num_samples) or (
        args.num_samples is None and time.time() - start < 10
    ):
        params = generator_configuration.sample_params()
        generator_configuration.sample_graph(args.num_nodes, **params)
        count += 1
    duration = time.time() - start
    print(
        f"generated {count} samples with {args.num_nodes} nodes in {duration:.3f} secs "
        f"({count / duration:.3f} per sec)"
    )


if __name__ == "__main__":
    __main__()  # pragma: no cover
