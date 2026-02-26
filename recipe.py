# type: ignore
import itertools as it
import os
from pathlib import Path

from cook import create_task, dict2args

from simulation_based_graph_inference.config import GENERATOR_CONFIGURATIONS


# Prevent each process from parallelizing which can lead to competition across processes.
os.environ.update(
    {
        "NUMEXPR_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
)


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
    # "latent_space_graph",
    "newman_watts_strogatz_graph",
]
ROOT = Path("workspace")

# Load basic configuration from the environment.
CONFIG = {
    "MAX_DEPTH": (int, 5),
    "NUM_SEEDS": (int, 5),
    "NUM_NODES": (int, 1000),
}
CONFIG = {
    key: type_(os.environ.get(key, default)) for key, (type_, default) in CONFIG.items()
}

# Set up different architecture specifications that we would like to compare.
DEPTHS = range(CONFIG["MAX_DEPTH"] + 1)
SEEDS = range(CONFIG["NUM_SEEDS"])

# This is the architecture we apply to all generators, not just the reference generators.
REFERENCE_ARCHITECTURES = {
    "conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-no_final_act-init_normal",
    "conv_8x2_res_scalar-dense_8x2_res_scalar_fixed_depth-pool_last-no_final_act-init_normal",
}
ARCHITECTURE_SPECIFICATIONS = {}
for depth in DEPTHS:
    # Create simple convolutional isomorphism layers with normalization for all but the first layer.
    conv = ["simple"] + ["norm"] * (depth - 1) if depth else ["none"]
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_simple_norm-dense_8x2-pool_concat-final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": "8,8",
        "conv": conv,
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_simple_norm-dense_8x4-pool_concat-final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": "8,8,8,8",
        "conv": conv,
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_simple_norm-dense_64x2-pool_concat-final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": "64,64",
        "conv": conv,
    }

    # Create convolutional isomorphism layers with two-layer dense networks after each layer.
    ARCHITECTURE_SPECIFICATIONS[
        ("conv_8x2-dense_8x2-pool_concat-final_act-init_normal", f"depth_{depth}")
    ] = {
        "dense": "8,8",
        "conv": ["8,8"] * depth if depth else ["none"],
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_dropout-dense_8x2-pool_concat-final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": "8,8",
        "conv": (["8,8"] * depth if depth else ["none"]) + ["dropout-0.5"],
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_identity-dense_8x2-pool_concat-final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": "8,8",
        "conv": ["res-identity-8,8"] * depth if depth else ["none"],
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2-pool_concat-final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": "8,8",
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2-pool_concat-final_act-init_small",
            f"depth_{depth}",
        )
    ] = {
        "dense": "8,8",
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
        "init-scale": 0.01,
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2-pool_concat-no_final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": "8,8",
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
        "final-activation": False,
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2-pool_concat-no_final_act-init_small",
            f"depth_{depth}",
        )
    ] = {
        "dense": "8,8",
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
        "init-scale": 0.01,
        "final-activation": False,
    }

    # Create architecture specification with last-layer pooling and depth-compensated dense layers.
    # This ensures fair comparison by keeping total layer count constant across different GNN depths.
    # Each GIN layer has a 2-layer MLP ("8,8"), so we compensate with 2-layer dense blocks.
    # Use residual connections around each dense block for consistency with GNN residual blocks.
    num_dense_blocks = (
        CONFIG["MAX_DEPTH"] - depth + 1
    )  # +1 for the baseline dense block ("8,8")
    dense_blocks = ["res-scalar-8,8"] * num_dense_blocks
    dense_spec = "_".join(dense_blocks)
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": dense_spec,
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
        "pooling": "last",
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-final_act-init_small",
            f"depth_{depth}",
        )
    ] = {
        "dense": dense_spec,
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
        "pooling": "last",
        "init-scale": 0.01,
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-no_final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": dense_spec,
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
        "pooling": "last",
        "final-activation": False,
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-no_final_act-init_small",
            f"depth_{depth}",
        )
    ] = {
        "dense": dense_spec,
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
        "pooling": "last",
        "init-scale": 0.01,
        "final-activation": False,
    }
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_res_scalar-dense_8x2_res_scalar_fixed_depth-pool_last-no_final_act-init_normal",
            f"depth_{depth}",
        )
    ] = {
        "dense": "res-scalar-8,8_res-scalar-8,8",
        "conv": ["res-scalar-8,8"] * depth if depth else ["none"],
        "pooling": "last",
        "final-activation": False,
    }
    ARCHITECTURE_SPECIFICATIONS[
        ("conv_16x2-dense_16x2-pool_concat-final_act-init_normal", f"depth_{depth}")
    ] = {
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
    ARCHITECTURE_SPECIFICATIONS[
        (
            "conv_8x2_dropout-dense_8x2-pool_concat-final_act-init_normal-with_clustering",
            f"depth_{depth}",
        )
    ] = {
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
    "validation": 1_000,
    "test": 1_000,
    "debug": 100,
}


# Collect tasks for groups (we'll create group tasks at the end)
reference_configuration_tasks = []
reference_architecture_tasks = []


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
            "python",
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
        create_task(
            f"{data_basename}:{split}",
            action=args,
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
        name = f"seed_{seed}"
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
        args = [
            "python",
            "-m",
            "simulation_based_graph_inference.scripts.train_nn",
        ] + dict2args(**kwargs)
        task = create_task(
            f"{basename}:{name}",
            action=args,
            targets=[target],
            dependencies=datasets,
        )
        if configuration in REFERENCE_CONFIGURATIONS:
            reference_configuration_tasks.append(task)
        if architecture in REFERENCE_ARCHITECTURES:
            reference_architecture_tasks.append(task)


# Inference for trees using a different method to compare with.
for config in ["gn_graph02", "gn_graph"]:
    test_data = ROOT / config / "data/test"
    target = ROOT / config / "cantwell/result.pkl"
    args = [
        "python",
        "-m",
        "simulation_based_graph_inference.scripts.infer_tree_kernel",
    ] + dict2args(test=test_data, result=target, config=config)
    create_task(
        f"{config}/cantwell",
        dependencies=[test_data / "meta.json"],
        targets=[target],
        action=args,
    )


# Profiling targets.
for configuration in GENERATOR_CONFIGURATIONS:
    basename = f"profile/{configuration}"
    target = ROOT / f"{basename}.prof"
    profile_script = Path("src/simulation_based_graph_inference/scripts/profile.py")
    args = [
        "python",
        "-m",
        "cProfile",
        "-o",
        str(target),
        str(profile_script),
    ] + dict2args(configuration=configuration, num_nodes=CONFIG["NUM_NODES"])
    create_task(
        f"{basename}:prof",
        targets=[target],
        action=args,
        dependencies=[profile_script],
    )

    target = ROOT / f"{basename}.lineprof"
    # Line profiling requires sequential commands, so use shell action
    action = (
        f"python -m kernprof -l -z -o {target}.tmp {profile_script} "
        f"--configuration={configuration} --num_nodes={CONFIG['NUM_NODES']} && "
        f"python -m line_profiler {target}.tmp > {target} && "
        f"rm -f {target}.tmp"
    )
    create_task(
        f"{basename}:lineprof",
        targets=[target],
        action=action,
        dependencies=[profile_script],
    )


# Create group tasks for convenience.
if reference_configuration_tasks:
    create_task(
        "reference_configurations",
        task_dependencies=reference_configuration_tasks,
    )

if reference_architecture_tasks:
    create_task(
        "reference_architecture",
        task_dependencies=reference_architecture_tasks,
    )
