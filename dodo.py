import doit_interface as di
from doit_interface import dict2args
import itertools as it
import os
import pathlib
from simulation_based_graph_inference.scripts.util import GENERATORS


ROOT = pathlib.Path("workspace")
DOIT_CONFIG = di.DOIT_CONFIG
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
    "NUM_SEEDS": (int, 5),
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

SPLITS = {
    "train": 100_000,
    "validation": 1_000,
    "test": 1_000,
}

for generator in GENERATORS:
    # Generate the data.
    datasets = []
    for seed, (split, num_samples) in enumerate(SPLITS.items()):
        data_basename = f"sinm2022/{generator}/data"
        directory = ROOT / data_basename / split
        target = directory / "length"
        datasets.append(target)
        args = ["$!", "-m", "simulation_based_graph_inference.scripts.sinm2022_data"] + \
            dict2args(seed=seed, generator=generator, num_samples=num_samples, directory=directory)
        manager(basename=data_basename, name=split, actions=[args], uptodate=[True],
                targets=[target])

    # Train the models.
    for seed, ((architecture, depth), specification) in it.product(SEEDS, SPECIFICATIONS.items()):
        name = f"seed-{seed}"
        basename = f"sinm2022/{generator}/{architecture}/{depth}"
        target = ROOT / f"{basename}/{name}.pkl"
        args = ["$!", "-m", "simulation_based_graph_inference.scripts.sinm2022_train"] + \
            dict2args(specification, {split: ROOT / data_basename / split for split in SPLITS},
                      seed=seed, generator=generator, result=target)
        manager(basename=basename, name=name, actions=[args], targets=[target], uptodate=[True],
                file_dep=datasets)
