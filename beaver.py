import beaver_build as bb
import itertools as it
import os
from simulation_based_graph_inference.scripts.util import GENERATORS

bb.Subprocess.ENV.update({
    "NUMEXPR_NUM_THREADS": 1,
    "OPENBLAS_NUM_THREADS": 1,
    "OMP_NUM_THREADS": 1,
})

CONFIG = {
    "MAX_DEPTH": (int, 5),
    "NUM_SEEDS": (int, 10),
}
CONFIG = {key: type(os.environ.get(key, default)) for key, (type, default) in CONFIG.items()}

DEPTHS = range(CONFIG["MAX_DEPTH"])
SEEDS = range(CONFIG["NUM_SEEDS"])
for generator, depth, seed in it.product(GENERATORS, DEPTHS, SEEDS):
    with bb.group_artifacts("workspace", "sinm2022", generator, f"depth_{depth}"):
        args = ["$!", "-m", "simulation_based_graph_inference.scripts.sinm2022", "--test=$@",
                f"--seed={seed}", generator, depth]
        bb.Subprocess(f"seed_{seed}.pkl", None, args)
