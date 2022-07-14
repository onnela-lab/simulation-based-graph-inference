from doit_interface import dict2args
import pickle
import pytest
from simulation_based_graph_inference.scripts import sinm2022_data, sinm2022_train, util


@pytest.mark.parametrize("generator", util.GENERATORS)
@pytest.mark.parametrize("dense", ["11,5", "7"])
@pytest.mark.parametrize("conv", ["none", "simple_norm_3_5,7"])
def test_sinm2022(generator: str, dense: str, conv: str, tmpwd: str):
    # Generate some data.
    steps_per_epoch = 7
    batch_size = 13
    args = dict2args(directory="data", generator=generator,
                     num_samples=steps_per_epoch * batch_size + 3)
    sinm2022_data.__main__(args)

    # Run the training.
    filename = "result.pkl"
    args = dict2args(
        patience=1, num_nodes=1, result=filename, batch_size=batch_size, generator=generator,
        seed=13, conv=conv, dense=dense, train="data", test="data", validation="data",
    )
    sinm2022_train.__main__(args)
    with open(filename, "rb") as fp:
        result = pickle.load(fp)

    expected_shape = (steps_per_epoch * batch_size + 3,)
    assert result["log_prob"].shape == expected_shape
    for key, dist in result["dists"].items():
        param = result["params"][key]
        assert dist.batch_shape == expected_shape
        assert param.shape[:1] == expected_shape
