import os
import pickle
import pytest
from simulation_based_graph_inference.scripts import sinm2022, util
import tempfile


@pytest.mark.parametrize("generator", util.GENERATORS)
@pytest.mark.parametrize("dense", ["11,5", "7"])
@pytest.mark.parametrize("conv", ["none", "simple_norm_3_5,7"])
def test_sinm2022(generator: str, dense: str, conv: str):
    steps_per_epoch = 7
    batch_size = 13

    with tempfile.TemporaryDirectory() as dir:
        filename = os.path.join(dir, "test.pkl")
        sinm2022.__main__([
            "--patience=1", "--num_nodes=10", f"--test={filename}", f"--batch_size={batch_size}",
            f"--steps_per_epoch={steps_per_epoch}", "--seed=13", generator, conv, dense,
        ])
        with open(filename, "rb") as fp:
            result = pickle.load(fp)

    expected_shape = (steps_per_epoch * batch_size,)
    assert result["log_prob"].shape == expected_shape
    for key, dist in result["dists"].items():
        param = result["params"][key]
        assert dist.batch_shape == expected_shape
        assert param.shape[:1] == expected_shape
