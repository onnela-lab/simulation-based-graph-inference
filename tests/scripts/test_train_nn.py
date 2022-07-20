from doit_interface import dict2args
import pickle
import pytest
from simulation_based_graph_inference import config
from simulation_based_graph_inference.scripts import generate_data, train_nn


@pytest.mark.parametrize("configuration", config.Configuration)
@pytest.mark.parametrize("dense", ["11,5", "7"])
@pytest.mark.parametrize("conv", ["none", "simple_norm_3_5,7"])
def test_sinm2022(configuration: config.Configuration, dense: str, conv: str, tmpwd: str):
    # Generate some data.
    steps_per_epoch = 7
    batch_size = 13
    num_batches = 11
    args = dict2args(directory="data", configuration=configuration.name, batch_size=batch_size,
                     num_batches=num_batches, num_nodes=10)
    generate_data.__main__(args)

    # Run the training.
    filename = "result.pkl"
    args = dict2args(
        patience=5, num_nodes=1, result=filename, batch_size=batch_size,
        configuration=configuration.name, seed=13, conv=conv, dense=dense, train="data",
        test="data", validation="data", steps_per_epoch=steps_per_epoch, max_num_epochs=3,
    )
    train_nn.__main__(args)
    with open(filename, "rb") as fp:
        result = pickle.load(fp)

    expected_shape = (num_batches * batch_size,)
    assert result["log_prob"].shape == expected_shape
    assert len(result["losses"]["train"]) == 3
    for key, dist in result["dists"].items():
        param = result["params"][key]
        assert dist.batch_shape == expected_shape
        assert param.shape[:1] == expected_shape
