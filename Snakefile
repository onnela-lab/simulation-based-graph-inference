# type: ignore
"""
Snakemake equivalent of recipe.py.

Usage (local):
    snakemake --cores 8 reference_configurations summary_training

Usage (SLURM):
    snakemake --cores 8 --executor slurm \
        --default-resources slurm_partition=<partition> mem_mb=4096 \
        reference_configurations summary_training

Environment variables:
    MAX_DEPTH   max GNN depth to sweep (default: 5)
    NUM_SEEDS   number of random seeds (default: 5)
    NUM_NODES   number of nodes per graph (default: 1000)
"""

import itertools as it
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_DEPTH = int(os.environ.get("MAX_DEPTH", 5))
NUM_SEEDS = int(os.environ.get("NUM_SEEDS", 5))
NUM_NODES = int(os.environ.get("NUM_NODES", 1000))

DEPTHS = list(range(MAX_DEPTH + 1))
SEEDS = list(range(NUM_SEEDS))

ROOT = "workspace"

SPLITS = ["train", "validation", "test", "debug"]
SPLIT_SIZES = {"train": 10_000, "validation": 1_000, "test": 1_000, "debug": 100}
BATCH_SIZE = 100

# Generator configurations (keys of GENERATOR_CONFIGURATIONS in config.py).
CONFIGURATIONS = [
    "poisson_random_attachment_graph",
    "random_connection_graph",
    "newman_watts_strogatz_graph",
    "redirection_graph",
    "copy_graph",
    "duplication_mutation_graph",
    "duplication_complementation_graph",
    "watts_strogatz_graph",
    "jackson_rogers_graph",
    "gn_graph",
    "gn_graph02",
]

REFERENCE_CONFIGURATIONS = [
    "duplication_complementation_graph",
    "watts_strogatz_graph",
    "localized_jackson_rogers_graph",
    "newman_watts_strogatz_graph",
]

REFERENCE_ARCHITECTURES = {
    "conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-no_final_act-init_normal",
    "conv_8x2_res_scalar-dense_8x2_res_scalar_fixed_depth-pool_last-no_final_act-init_normal",
}

# ---------------------------------------------------------------------------
# Build architecture specifications matching recipe.py exactly
# ---------------------------------------------------------------------------

ARCH_SPECS = {}  # (architecture, depth_label) -> kwargs dict

for depth in DEPTHS:
    conv_simple_norm = ["simple"] + ["norm"] * (depth - 1) if depth else ["none"]

    ARCH_SPECS[("conv_simple_norm-dense_8x2-pool_concat-final_act-init_normal", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_simple_norm,
    }
    ARCH_SPECS[("conv_simple_norm-dense_8x4-pool_concat-final_act-init_normal", f"depth_{depth}")] = {
        "dense": "8,8,8,8", "conv": conv_simple_norm,
    }
    ARCH_SPECS[("conv_simple_norm-dense_64x2-pool_concat-final_act-init_normal", f"depth_{depth}")] = {
        "dense": "64,64", "conv": conv_simple_norm,
    }

    conv_8x2      = ["8,8"] * depth if depth else ["none"]
    conv_8x2_drop = (["8,8"] * depth if depth else ["none"]) + ["dropout-0.5"]
    conv_res_id   = ["res-identity-8,8"] * depth if depth else ["none"]
    conv_res_sc   = ["res-scalar-8,8"] * depth if depth else ["none"]

    ARCH_SPECS[("conv_8x2-dense_8x2-pool_concat-final_act-init_normal", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_8x2,
    }
    ARCH_SPECS[("conv_8x2_dropout-dense_8x2-pool_concat-final_act-init_normal", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_8x2_drop,
    }
    ARCH_SPECS[("conv_8x2_res_identity-dense_8x2-pool_concat-final_act-init_normal", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_res_id,
    }
    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2-pool_concat-final_act-init_normal", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_res_sc,
    }
    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2-pool_concat-final_act-init_small", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_res_sc, "init-scale": "0.01",
    }
    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2-pool_concat-no_final_act-init_normal", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_res_sc, "final-activation": "False",
    }
    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2-pool_concat-no_final_act-init_small", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_res_sc, "init-scale": "0.01", "final-activation": "False",
    }

    # Depth-compensated last-layer pooling variants.
    num_dense_blocks = MAX_DEPTH - depth + 1
    dense_spec = "_".join(["res-scalar-8,8"] * num_dense_blocks)

    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-final_act-init_normal", f"depth_{depth}")] = {
        "dense": dense_spec, "conv": conv_res_sc, "pooling": "last",
    }
    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-final_act-init_small", f"depth_{depth}")] = {
        "dense": dense_spec, "conv": conv_res_sc, "pooling": "last", "init-scale": "0.01",
    }
    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-no_final_act-init_normal", f"depth_{depth}")] = {
        "dense": dense_spec, "conv": conv_res_sc, "pooling": "last", "final-activation": "False",
    }
    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2_res_scalar_comp_depth-pool_last-no_final_act-init_small", f"depth_{depth}")] = {
        "dense": dense_spec, "conv": conv_res_sc, "pooling": "last", "init-scale": "0.01", "final-activation": "False",
    }
    ARCH_SPECS[("conv_8x2_res_scalar-dense_8x2_res_scalar_fixed_depth-pool_last-no_final_act-init_normal", f"depth_{depth}")] = {
        "dense": "res-scalar-8,8_res-scalar-8,8", "conv": conv_res_sc, "pooling": "last", "final-activation": "False",
    }
    ARCH_SPECS[("conv_16x2-dense_16x2-pool_concat-final_act-init_normal", f"depth_{depth}")] = {
        "dense": "16,16", "conv": ["16,16"] * depth if depth else ["none"],
    }

    # With-clustering variant.
    if depth:
        conv_clust = ["8,8"] * depth
        if depth > 1:
            conv_clust.insert(2, "insert-clustering")
        conv_clust.append("dropout-0.5")
    else:
        conv_clust = ["none"]
    ARCH_SPECS[("conv_8x2_dropout-dense_8x2-pool_concat-final_act-init_normal-with_clustering", f"depth_{depth}")] = {
        "dense": "8,8", "conv": conv_clust,
    }

# Flatten conv lists to underscore-separated strings (mirrors recipe.py).
for spec in ARCH_SPECS.values():
    spec["conv"] = "_".join(spec["conv"])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def arch_cmd_extra(spec):
    """Return extra CLI flags beyond --conv / --dense for a given arch spec."""
    flags = []
    if "pooling" in spec:
        flags += [f"--pooling={spec['pooling']}"]
    if "init-scale" in spec:
        flags += [f"--init-scale={spec['init-scale']}"]
    if "final-activation" in spec:
        flags += [f"--final-activation={spec['final-activation']}"]
    return " ".join(flags)


# Pre-compute which (configuration, architecture, depth, seed) combos are active,
# mirroring the `continue` logic in recipe.py.
ACTIVE_NN_JOBS = [
    (config, arch, depth, seed)
    for config in CONFIGURATIONS
    for (arch, depth), spec in ARCH_SPECS.items()
    for seed in SEEDS
    if arch in REFERENCE_ARCHITECTURES or config in REFERENCE_CONFIGURATIONS
]

# Summary dense specs.
SUMMARY_DENSE_SPECS = {
    "dense_8x2_res_scalar": "res-scalar-8,8_res-scalar-8,8",
}

# ---------------------------------------------------------------------------
# Default target: everything needed for reference_configurations +
# summary_training (mirrors the two group tasks in recipe.py).
# ---------------------------------------------------------------------------

ALL_NN_TARGETS = [
    ROOT + f"/{config}/{arch}/{depth}/seed_{seed}.pkl"
    for (config, arch, depth, seed) in ACTIVE_NN_JOBS
    if config in REFERENCE_CONFIGURATIONS
]
ALL_SUMMARY_TARGETS = expand(
    ROOT + "/{config}/summary/{dense_name}/seed_{seed}.pkl",
    config=CONFIGURATIONS,
    dense_name=list(SUMMARY_DENSE_SPECS.keys()),
    seed=SEEDS,
)

rule all:
    input:
        ALL_NN_TARGETS + ALL_SUMMARY_TARGETS,


# ---------------------------------------------------------------------------
# Rule 1: generate_data
# ---------------------------------------------------------------------------

rule generate_data:
    output:
        ROOT + "/{config}/data/{split}/meta.json",
    params:
        seed=lambda wc: SPLITS.index(wc.split),
        num_batches=lambda wc: SPLIT_SIZES[wc.split] // BATCH_SIZE,
    resources:
        mem_mb=4000,
        runtime=90,
        slurm_partition="serial_requeue",
    shell:
        """
        python -m simulation_based_graph_inference.scripts.generate_data \
            --seed={params.seed} \
            --configuration={wildcards.config} \
            --batch_size={BATCH_SIZE} \
            --directory={ROOT}/{wildcards.config}/data/{wildcards.split} \
            --num_batches={params.num_batches} \
            --num_nodes={NUM_NODES}
        """


# ---------------------------------------------------------------------------
# Rule 2: train_nn
# (Snakemake wildcards can't encode the arch spec, so we use a checkpoint-style
# approach: generate one rule per active job via a helper function.)
# ---------------------------------------------------------------------------

def _nn_input_datasets(wildcards):
    return expand(
        ROOT + "/" + wildcards.config + "/data/{split}/meta.json",
        split=[s for s in SPLITS if s != "debug"],
    )


rule train_nn:
    input:
        _nn_input_datasets,
    output:
        ROOT + "/{config}/{arch}/{depth}/seed_{seed}.pkl",
    params:
        spec=lambda wc: ARCH_SPECS.get((wc.arch, wc.depth), {}),
        extra=lambda wc: arch_cmd_extra(ARCH_SPECS.get((wc.arch, wc.depth), {})),
    resources:
        mem_mb=8000,
        runtime=480,
    shell:
        """
        python -m simulation_based_graph_inference.scripts.train_nn \
            --seed={wildcards.seed} \
            --configuration={wildcards.config} \
            --conv={params.spec[conv]} \
            --dense={params.spec[dense]} \
            --result={output} \
            --train={ROOT}/{wildcards.config}/data/train \
            --validation={ROOT}/{wildcards.config}/data/validation \
            --test={ROOT}/{wildcards.config}/data/test \
            {params.extra}
        """


# ---------------------------------------------------------------------------
# Rule 3: generate_summaries
# ---------------------------------------------------------------------------

rule generate_summaries:
    input:
        ROOT + "/{config}/data/{split}/meta.json",
    output:
        ROOT + "/{config}/summaries/{split}.pkl",
    resources:
        mem_mb=4000,
        runtime=60,
        slurm_partition="serial_requeue",
    shell:
        """
        python -m simulation_based_graph_inference.scripts.generate_summaries \
            {ROOT}/{wildcards.config}/data/{wildcards.split} \
            {output} \
            --configuration={wildcards.config}
        """


# ---------------------------------------------------------------------------
# Rule 4: train_summary
# ---------------------------------------------------------------------------

rule train_summary:
    input:
        train=ROOT + "/{config}/summaries/train.pkl",
        validation=ROOT + "/{config}/summaries/validation.pkl",
        test=ROOT + "/{config}/summaries/test.pkl",
    output:
        ROOT + "/{config}/summary/{dense_name}/seed_{seed}.pkl",
    params:
        dense_spec=lambda wc: SUMMARY_DENSE_SPECS[wc.dense_name],
    resources:
        mem_mb=4000,
        runtime=120,
    shell:
        """
        python -m simulation_based_graph_inference.scripts.train_summary \
            --seed={wildcards.seed} \
            --configuration={wildcards.config} \
            --dense={params.dense_spec} \
            --result={output} \
            --train={input.train} \
            --validation={input.validation} \
            --test={input.test}
        """


# ---------------------------------------------------------------------------
# Rule 5: infer_tree_kernel (cantwell baseline)
# ---------------------------------------------------------------------------

rule infer_tree_kernel:
    input:
        ROOT + "/{config}/data/test/meta.json",
    output:
        ROOT + "/{config}/cantwell/result.pkl",
    wildcard_constraints:
        config="gn_graph|gn_graph02",
    shell:
        """
        python -m simulation_based_graph_inference.scripts.infer_tree_kernel \
            --test={ROOT}/{wildcards.config}/data/test \
            --result={output} \
            --config={wildcards.config}
        """


# ---------------------------------------------------------------------------
# Convenience group targets (mirrors cook group tasks)
# ---------------------------------------------------------------------------

rule reference_configurations:
    """All NN training jobs for the reference generator configurations."""
    input:
        [
            ROOT + f"/{config}/{arch}/{depth}/seed_{seed}.pkl"
            for (config, arch, depth, seed) in ACTIVE_NN_JOBS
            if config in REFERENCE_CONFIGURATIONS
        ],


rule reference_architecture:
    """All NN training jobs for the reference architectures (across all generators)."""
    input:
        [
            ROOT + f"/{config}/{arch}/{depth}/seed_{seed}.pkl"
            for (config, arch, depth, seed) in ACTIVE_NN_JOBS
            if arch in REFERENCE_ARCHITECTURES
        ],


rule summary_training:
    """All summary-based density estimator training jobs."""
    input:
        expand(
            ROOT + "/{config}/summary/{dense_name}/seed_{seed}.pkl",
            config=CONFIGURATIONS,
            dense_name=SUMMARY_DENSE_SPECS.keys(),
            seed=SEEDS,
        ),
