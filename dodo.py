import doit_interface as di
from doit_interface import dict2args
import itertools as it
import os
import pathlib
from simulation_based_graph_inference.config import Configuration


# Generator configurations for which we run all architectures. This should include one from each
# "class" of models, such as duplication divergence models or spatial graphs. We will use these
# reference generators to pick good architectures. Running all architectures on all generators would
# be too computationally intensive.
REFERENCE_CONFIGURATIONS = [
    Configuration.duplication_complementation_graph, Configuration.random_geometric_graph,
    Configuration.watts_strogatz_graph, Configuration.jackson_rogers_graph,
    Configuration.degree_attachment_graph,
]
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
    "NUM_SEEDS": (int, 3),
}
CONFIG = {key: type(os.environ.get(key, default)) for key, (type, default) in CONFIG.items()}

# Set up different architecture specifications that we would like to compare.
DEPTHS = range(CONFIG["MAX_DEPTH"] + 1)
SEEDS = range(CONFIG["NUM_SEEDS"])

# This is the architecture we apply to all generators, not just the reference generators.
REFERENCE_ARCHITECTURE = "simple-narrow"
ARCHITECTURE_SPECIFICATIONS = {}
for depth in DEPTHS:
    # Create simple convolutional isomorphism layers with normalization for all but the first layer.
    conv = "_".join(["simple"] + ["norm"] * (depth - 1)) if depth else "none"
    ARCHITECTURE_SPECIFICATIONS[("simple-narrow", f"depth-{depth}")] = {
        "dense": "8,8",
        "conv": conv,
    }
    ARCHITECTURE_SPECIFICATIONS[("simple-deep", f"depth-{depth}")] = {
        "dense": "8,8,8,8",
        "conv": conv,
    }
    ARCHITECTURE_SPECIFICATIONS[("simple-wide", f"depth-{depth}")] = {
        "dense": "64,64",
        "conv": conv,
    }

    # Create convolutional isomorphism layers with two-layer dense networks after each layer.
    ARCHITECTURE_SPECIFICATIONS[("gin-narrow", f"depth-{depth}")] = {
        "dense": "8,8",
        "conv": "_".join(["8,8"] * depth) if depth else "none",
    }
    ARCHITECTURE_SPECIFICATIONS[("gin-medium", f"depth-{depth}")] = {
        "dense": "16,16",
        "conv": "_".join(["16,16"] * depth) if depth else "none",
    }

BATCH_SIZE = 100
SPLITS = {
    "train": 10_000,
    "validation": 1_000,
    "test": 1_000,
}

reference_configurations = di.group_tasks("reference_configurations")

for configuration in Configuration:
    # Generate the data.
    datasets = []
    for seed, (split, num_samples) in enumerate(SPLITS.items()):
        assert num_samples % BATCH_SIZE == 0, "number of samples must be a multiple of BATCH_SIZE"
        data_basename = f"{configuration.name}/data"
        directory = ROOT / data_basename / split
        target = directory / "meta.json"
        datasets.append(target)
        args = ["$!", "-m", "simulation_based_graph_inference.scripts.generate_data"] + \
            dict2args(seed=seed, configuration=configuration.name, batch_size=BATCH_SIZE,
                      directory=directory, num_batches=num_samples // BATCH_SIZE)
        manager(basename=data_basename, name=split, actions=[args], uptodate=[True],
                targets=[target])

    # Train the models.
    for seed, ((architecture, depth), specification) in \
            it.product(SEEDS, ARCHITECTURE_SPECIFICATIONS.items()):
        # Don't run the training if this is not the reference architecture or a reference generator.
        if architecture != REFERENCE_ARCHITECTURE and configuration not in REFERENCE_CONFIGURATIONS:
            continue
        name = f"seed-{seed}"
        basename = f"{configuration.name}/{architecture}/{depth}"
        target = ROOT / f"{basename}/{name}.pkl"
        args = ["$!", "-m", "simulation_based_graph_inference.scripts.train_nn"] + \
            dict2args(specification, {split: ROOT / data_basename / split for split in SPLITS},
                      seed=seed, configuration=configuration.name, result=target)
        task = manager(basename=basename, name=name, actions=[args], targets=[target],
                       uptodate=[True], file_dep=datasets)
        if configuration in REFERENCE_CONFIGURATIONS:
            reference_configurations(task)


# Profiling targets.
for configuration in Configuration:
    basename = f"profile/{configuration.name}"
    target = ROOT / f"{basename}.prof"
    args = ["$!", "-m", "cProfile", "-o", "$@", "$^"] + dict2args(configuration=configuration.name)
    manager(basename=basename, name="prof", targets=[target], actions=[args],
            file_dep=["simulation_based_graph_inference/scripts/profile.py"])

    target = ROOT / f"{basename}.lineprof"
    actions = [
        f"$! -m kernprof -l -z -o $@.tmp $^ --configuration={configuration.name}",
        "$! -m line_profiler $@.tmp > $@",
        "rm -f $@.tmp",
    ]
    manager(basename=basename, name="lineprof", targets=[target], actions=actions,
            file_dep=["simulation_based_graph_inference/scripts/profile.py"])
