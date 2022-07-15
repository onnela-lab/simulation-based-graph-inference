import time
from simulation_based_graph_inference import generators  # Absolute import for line_profiler CLI.
from simulation_based_graph_inference import models
from simulation_based_graph_inference.scripts import util


def __main__(args: list[str] = None):
    parser = util.get_parser(10_000)
    parser.add_argument("--num_samples", "-m", help="number of independent graph samples", type=int)
    args = parser.parse_args(args)

    # Generate parameters for each method.
    generator = getattr(generators, args.generator)
    prior = models.get_prior(generator)

    # Set up line profiling if desired.
    try:
        generator = profile(generator)  # pyright: reportUndefinedVariable=false
    except NameError as ex:
        if str(ex) != "name 'profile' is not defined":
            raise  # pragma: no cover

    # Generate samples.
    start = time.time()
    count = 0
    while (args.num_samples and count < args.num_samples) \
            or (args.num_samples is None and time.time() - start < 10):
        params = {key: value.sample() for key, value in prior.items()}
        generator(args.num_nodes, **params)
        count += 1
    duration = time.time() - start
    print(f"generated {count} samples in {duration:.3f} secs ({count / duration:.3f} per sec)")


if __name__ == "__main__":
    __main__()  # pragma: no cover
