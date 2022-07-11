import doit_interface as di
import itertools as it
import os
from simulation_based_graph_inference.scripts.util import GENERATORS

manager = di.Manager.get_instance()

# Prevent each process from parallelizing which can lead to competition across processes.
di.SubprocessAction.set_global_env({
    "NUMEXPR_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 1,
    "OMP_NUM_THREADS": 1,
})

# Load basic configuration from the environment.
CONFIG = {
    "MAX_DEPTH": (int, 4),
    "NUM_SEEDS": (int, 10),
}
CONFIG = {key: type(os.environ.get(key, default)) for key, (type, default) in CONFIG.items()}

# Set up different architecture specifications that we would like to compare.
DEPTHS = range(CONFIG["MAX_DEPTH"] + 1)
SEEDS = range(CONFIG["NUM_SEEDS"])

SPECIFICATIONS = {}
for depth in DEPTHS:
    # Create simple convolutional isomorphism layers with normalization for all but the first layer.
    conv = "_".join(["simple"] + ["norm"] * (depth - 1)) if depth else "none"
    SPECIFICATIONS[("simple-narrow", f"depth-{depth}")] = {
        "dense": "8,8",
        "conv": conv,
    }
    SPECIFICATIONS[("simple-deep", f"depth-{depth}")] = {
        "dense": "8,8,8,8",
        "conv": conv,
    }
    SPECIFICATIONS[("simple-wide", f"depth-{depth}")] = {
        "dense": "64,64",
        "conv": conv,
    }

    # Create convolutional isomorphism layers with two-layer dense networks after each layer.
    SPECIFICATIONS[("gin-narrow", f"depth-{depth}")] = {
        "dense": "8,8",
        "conv": "_".join(["8,8"] * depth) if depth else "none",
    }
    SPECIFICATIONS[("gin-medium", f"depth-{depth}")] = {
        "dense": "16,16",
        "conv": "_".join(["16,16"] * depth) if depth else "none",
    }

# Generate targets for the different configurations.
for generator, seed, (key, specification) in it.product(GENERATORS, SEEDS, SPECIFICATIONS.items()):
    args = ["$!", "-m", "simulation_based_graph_inference.scripts.sinm2022", "--test=$@",
            f"--seed={seed}", generator, specification["conv"], specification["dense"]]
    name = f"seed-{seed}"
    basename = f"sinm2022/{generator}/{key[0]}/{key[1]}"
    target = f"workspace/{basename}/{name}.pkl"
    manager(basename=basename, name=name, actions=[args], targets=[target], uptodate=[True])
