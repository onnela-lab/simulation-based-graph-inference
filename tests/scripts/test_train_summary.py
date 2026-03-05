from cook import dict2args
import pickle
import pytest
from simulation_based_graph_inference import config
from simulation_based_graph_inference.scripts import (
    generate_data,
    generate_summaries,
    train_summary,
)


def _check_result(filename: str, num_samples: int) -> dict:
    with open(filename, "rb") as fp:
        result = pickle.load(fp)

    expected_shape = (num_samples,)
    assert result["log_prob"].shape == expected_shape
    assert len(result["losses"]["train"]) == 3
    for key, dist in result["dists"].items():
        param = result["params"][key]
        assert dist.batch_shape == expected_shape
        assert param.shape == expected_shape
    assert result["features"].isfinite().all()
    return result


@pytest.mark.parametrize("configuration", config.GENERATOR_CONFIGURATIONS)
@pytest.mark.parametrize("dense", ["11,5", "7"])
def test_train_summary(configuration: str, dense: str, tmpwd: str) -> None:
    # Generate some data. Use larger graphs (50 nodes) to avoid NaN issues in
    # summary statistics that can occur with very small/sparse graphs.
    batch_size = 13
    num_batches = 11
    args = dict2args(
        directory="data",
        configuration=configuration,
        batch_size=batch_size,
        num_batches=num_batches,
        num_nodes=50,
    )
    generate_data.__main__(args)

    # Generate summaries.
    summaries_file = "summaries.pkl"
    args = ["data", summaries_file, "--configuration", configuration]
    generate_summaries.__main__(args)

    # Run the training.
    filename = "result.pkl"
    args = dict(
        patience=5,
        result=filename,
        batch_size=batch_size,
        configuration=configuration,
        seed=13,
        dense=dense,
        train=summaries_file,
        validation=summaries_file,
        test=summaries_file,
        max_num_epochs=3,
    )
    train_summary.__main__(dict2args(**args))  # type: ignore[arg-type]
    num_samples = batch_size * num_batches
    _check_result(filename, num_samples)

    # Apply transfer learning using the dense layers from a previous result.
    filename = "transfer_result.pkl"
    args.update(dense="file:result.pkl", result=filename)
    train_summary.__main__(dict2args(**args))  # type: ignore[arg-type]
    result = _check_result(filename, num_samples)
    assert result["dense"] == "file:result.pkl"
