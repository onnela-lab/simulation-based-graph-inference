from doit_interface import dict2args
import numpy as np
import pickle
import pytest
import random
from simulation_based_graph_inference import config
from simulation_based_graph_inference.scripts import generate_data, train_nn


def _check_result(filename: str, num_batches: int, batch_size: int) -> None:
    with open(filename, "rb") as fp:
        result = pickle.load(fp)

    expected_shape = (num_batches * batch_size,)
    assert result["log_prob"].shape == expected_shape
    assert len(result["losses"]["train"]) == 3
    for key, dist in result["dists"].items():
        param = result["params"][key]
        assert dist.batch_shape == expected_shape
        assert param.shape[:1] == expected_shape
    np.testing.assert_array_less(np.abs(result["features"]), 1)
    return result


@pytest.mark.parametrize("configuration", config.GENERATOR_CONFIGURATIONS)
# We only test transfer learning on a few models to avoid combinatorial explosion in testing.
@pytest.mark.parametrize("transfer_configuration",
                         random.sample(list(config.GENERATOR_CONFIGURATIONS), 3))
@pytest.mark.parametrize("dense", ["11,5", "7"])
@pytest.mark.parametrize("conv", ["none", "simple_norm_3_5,7"])
def test_train_nn(configuration: str, transfer_configuration: str, dense: str, conv: str,
                  tmpwd: str) -> None:
    # Generate some data.
    steps_per_epoch = 7
    batch_size = 13
    num_batches = 11
    args = dict2args(directory="data", configuration=configuration, batch_size=batch_size,
                     num_batches=num_batches, num_nodes=10)
    generate_data.__main__(args)

    # Run the training.
    filename = "result.pkl"
    args = dict(
        patience=5, num_nodes=1, result=filename, batch_size=batch_size,
        configuration=configuration, seed=13, conv=conv, dense=dense, train="data",
        test="data", steps_per_epoch=steps_per_epoch, max_num_epochs=3,
    )
    train_nn.__main__(dict2args(args))
    _check_result(filename, batch_size, num_batches)

    # Apply transfer learning using the other configuration.
    filename = "transfer_result.pkl"
    args.update(conv="file:result.pkl", dense="file:result.pkl", result=filename)
    train_nn.__main__(dict2args(args))
    result = _check_result(filename, batch_size, num_batches)
    assert result["dense"] == "file:result.pkl"
    assert result["conv"] == "file:result.pkl"
