# type: ignore
import doit_interface as di
from doit_interface import dict2args
import itertools as it
import os
import pathlib
from simulation_based_graph_inference.config import GENERATOR_CONFIGURATIONS


# Generator configurations for which we run all architectures. This should include one from each
# "class" of models, such as duplication divergence models or spatial graphs. We will use these
# reference generators to pick good architectures. Running all architectures on all generators would
# be too computationally intensive.
REFERENCE_CONFIGURATIONS = [
    "duplication_complementation_graph",
    # "random_geometric_graph",
    "watts_strogatz_graph",
    "localized_jackson_rogers_graph",
    # "degree_attachment_graph",
    "latent_space_graph",
    "newman_watts_strogatz_graph",
]
ROOT = pathlib.Path("workspace")
DOIT_CONFIG = di.DOIT_CONFIG
manager = di.Manager.get_instance()

# Prevent each process from parallelizing which can lead to competition across processes.
di.SubprocessAction.set_global_env(
    {
        "NUMEXPR_NUM_THREADS": 1,
        "OPENBLAS_NUM_THREADS": 1,
        "OMP_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
    }
)

# Load basic configuration from the environment.
CONFIG = {
    "MAX_DEPTH": (int, 5),
    "NUM_SEEDS": (int, 3),
    "NUM_NODES": (int, 1000),
}
CONFIG = {
    key: type(os.environ.get(key, default)) for key, (type, default) in CONFIG.items()
}

# Set up different architecture specifications that we would like to compare.
DEPTHS = range(CONFIG["MAX_DEPTH"] + 1)
SEEDS = range(CONFIG["NUM_SEEDS"])

# This is the architecture we apply to all generators, not just the reference generators.
REFERENCE_ARCHITECTURES = {
    # "residual-identity-gin-narrow",
    # "gin-narrow",
    "residual-scalar-gin-narrow",
}
ARCHITECTURE_SPECIFICATIONS = {}
for depth in DEPTHS:
    # Create simple convolutional isomorphism layers with normalization for all but the first layer.
    conv = ["simple"] + ["norm"] * (depth - 1) if depth else ["none"]
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
        "conv": ["8,8"] * depth if depth else ["none"],
    }
    ARCHITECTURE_SPECIFICATIONS[("gin-narrow-dropout", f"depth-{depth}")] = {
        "dense": "8,8",
        "conv": (["8,8"] * depth if depth else ["none"]) + ["dropout-0.5"],
    }
    ARCHITECTURE_SPECIFICATIONS[("residual-identity-gin-narrow", f"depth-{depth}")] = {
        "dense": "8,8",
        "conv": ["res-identity-8,8"] * depth if depth else ["none"],
    }
    ARCHITECTURE_SPECIFICATIONS[("residual-scalar-gin-narrow", f"depth-{depth}")] = {
        "dense": "8,8",
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
    }
    ARCHITECTURE_SPECIFICATIONS[("gin-medium", f"depth-{depth}")] = {
        "dense": "16,16",
        "conv": ["16,16"] * depth if depth else ["none"],
    }

    # Graph isomorphism networks with an insertion of the clustering coefficient after the second
    # layer.
    if depth:
        conv = ["8,8"] * depth
        if depth > 1:
            conv.insert(2, "insert-clustering")
        conv.append("dropout-0.5")
    else:
        conv = ["none"]
    ARCHITECTURE_SPECIFICATIONS[("gin-narrow-clustering", f"depth-{depth}")] = {
        # We may want slightly deeper dense network if we're injecting the clustering because it's
        # an "engineered" feature.
        "dense": "8,8",
        "conv": conv,
    }
for arch in ARCHITECTURE_SPECIFICATIONS.values():
    arch["conv"] = "_".join(arch["conv"])

BATCH_SIZE = 100
SPLITS = {
    "train": 10_000,
    "test": 1_000,
    "debug": 100,
}

reference_configurations = di.group_tasks("reference_configurations")
reference_architecture = di.group_tasks("reference_architecture")
transfer_learning = di.group_tasks("transfer_learning")

for configuration in GENERATOR_CONFIGURATIONS:
    # Generate the data.
    datasets = []
    for seed, (split, num_samples) in enumerate(SPLITS.items()):
        assert num_samples % BATCH_SIZE == 0, (
            "number of samples must be a multiple of BATCH_SIZE"
        )
        data_basename = f"{configuration}/data"
        directory = ROOT / data_basename / split
        target = directory / "meta.json"
        datasets.append(target)
        args = [
            "$!",
            "-m",
            "simulation_based_graph_inference.scripts.generate_data",
        ] + dict2args(
            seed=seed,
            configuration=configuration,
            batch_size=BATCH_SIZE,
            directory=directory,
            num_batches=num_samples // BATCH_SIZE,
            num_nodes=CONFIG["NUM_NODES"],
        )
        manager(
            basename=data_basename,
            name=split,
            actions=[args],
            uptodate=[True],
            targets=[target],
        )

    # Train the models.
    for seed, ((architecture, depth), specification) in it.product(
        SEEDS, ARCHITECTURE_SPECIFICATIONS.items()
    ):
        # Don't run the training if this is not the reference architecture or a reference generator.
        if (
            architecture not in REFERENCE_ARCHITECTURES
            and configuration not in REFERENCE_CONFIGURATIONS
        ):
            continue
        name = f"seed-{seed}"
        basename = f"{configuration}/{architecture}/{depth}"
        target = ROOT / f"{basename}/{name}.pkl"
        kwargs = (
            specification
            | {"seed": seed, "configuration": configuration, "result": target}
            | {
                split: ROOT / data_basename / split
                for split in SPLITS
                if split != "debug"
            }
        )
        args = ["$!", "-m", "simulation_based_graph_inference.scripts.train_nn"]
        task = manager(
            basename=basename,
            name=name,
            actions=[args + dict2args(kwargs)],
            targets=[target],
            uptodate=[True],
            file_dep=datasets,
        )
        if configuration in REFERENCE_CONFIGURATIONS:
            reference_configurations(task)
        if architecture in REFERENCE_ARCHITECTURES:
            reference_architecture(task)

        # NOTE: Do not run transfer learning.
        continue

        # Skip transfer learning if this is not the reference configuration.
        if architecture not in REFERENCE_ARCHITECTURES:
            continue

        # Run transfer learning for this configuration given features extractetd from all other
        # models. This may seem reversed from what we'd actually do in terms of procedural
        # execution, but we're only declaring tasks here. I.e., `transfer_configuration` is the
        # model that extracts the features.
        for transfer_configuration in GENERATOR_CONFIGURATIONS:
            # Don't need to do transfer learning if the two configurations are the same.
            if transfer_configuration == configuration:
                continue
            other_basename = f"{transfer_configuration}/{architecture}/{depth}"
            other_target = ROOT / f"{other_basename}/{name}.pkl"
            kwargs["conv"] = f"file:{other_target}"
            kwargs["dense"] = f"file:{other_target}"

            transfer_basename = (
                f"{configuration}/transfer/{transfer_configuration}/"
                f"{architecture}/{depth}"
            )
            transfer_target = ROOT / f"{transfer_basename}/{name}.pkl"
            kwargs["result"] = transfer_target
            task = manager(
                basename=transfer_basename,
                name=name,
                actions=[args + dict2args(kwargs)],
                targets=[transfer_target],
                file_dep=datasets + [other_target],
            )
            transfer_learning(task)


# Inference for trees using a different method to compare with.
config = "gn_graph"
test_data = ROOT / config / "data/test"
target = ROOT / config / "cantwell/result.pkl"
args = [
    "$!",
    "-m",
    "simulation_based_graph_inference.scripts.infer_tree_kernel",
] + dict2args(test=test_data, result=target)
manager(
    basename=f"{config}/cantwell",
    file_dep=[test_data / "meta.json"],
    targets=[target],
    actions=[args],
)


# Monte Carlo sampling for the latent space model.
config = "latent_space_graph"
test_data = ROOT / config / "data/test"
target = ROOT / config / "mcmc/result.pkl"
args = [
    "$!",
    "-m",
    "simulation_based_graph_inference.scripts.infer_latent_space_params",
] + dict2args(test=test_data, result=target)
manager(
    basename=f"{config}/mcmc",
    file_dep=[test_data / "meta.json"],
    targets=[target],
    actions=[args],
)


# Profiling targets.
for configuration in GENERATOR_CONFIGURATIONS:
    basename = f"profile/{configuration}"
    target = ROOT / f"{basename}.prof"
    args = ["$!", "-m", "cProfile", "-o", "$@", "$^"] + dict2args(
        configuration=configuration, num_nodes=CONFIG["NUM_NODES"]
    )
    manager(
        basename=basename,
        name="prof",
        targets=[target],
        actions=[args],
        file_dep=["src/simulation_based_graph_inference/scripts/profile.py"],
    )

    target = ROOT / f"{basename}.lineprof"
    actions = [
        f"$! -m kernprof -l -z -o $@.tmp $^ --configuration={configuration} "
        f"--num_nodes={CONFIG['NUM_NODES']}",  # noqa: E131
        "$! -m line_profiler $@.tmp > $@",
        "rm -f $@.tmp",
    ]
    manager(
        basename=basename,
        name="lineprof",
        targets=[target],
        actions=actions,
        file_dep=["simulation_based_graph_inference/scripts/profile.py"],
    )
