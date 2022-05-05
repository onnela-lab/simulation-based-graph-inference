import os
import pickle
import pytest
from simulation_based_graph_inference.scripts import sinm2022
import tempfile


@pytest.mark.parametrize("generator", sinm2022.GENERATORS)
@pytest.mark.parametrize("num_layers", [0, 2])
def test_profile(generator: str, num_layers: int):
    steps_per_epoch = 7
    batch_size = 13

    with tempfile.TemporaryDirectory() as dir:
        filename = os.path.join(dir, "test.pkl")
        sinm2022.__main__([
            "--patience=1", "--num_nodes=10", f"--test={filename}", f"--batch_size={batch_size}",
            f"--steps_per_epoch={steps_per_epoch}", "--seed=13", generator, str(num_layers)
        ])
        with open(filename, "rb") as fp:
            result = pickle.load(fp)

    expected_shape = (steps_per_epoch * batch_size,)
    assert result["log_prob"].shape == expected_shape
    for key, dist in result["dists"].items():
        param = result["params"][key]
        assert dist.batch_shape == expected_shape
        assert param.shape[:1] == expected_shape
