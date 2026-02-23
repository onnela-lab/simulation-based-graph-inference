from cook import dict2args
import os
import pickle
import pytest
from scipy import stats
from simulation_based_graph_inference.scripts import generate_data, infer_tree_kernel


@pytest.mark.filterwarnings("ignore:divide by zero encountered in power")
def test_infer_tree_kernel(tmpwd: str):
    n = 10
    datadir = os.path.join(tmpwd, "data")
    generate_data.__main__(
        dict2args(
            num_nodes=100,
            configuration="gn_graph",
            batch_size=n,
            num_batches=1,
            directory=datadir,
        )
    )
    filename = os.path.join(tmpwd, "result.pkl")
    infer_tree_kernel.__main__(dict2args(test=datadir, result=filename))

    with open(filename, "rb") as fp:
        result = pickle.load(fp)
    for key in ["log_prob", "gamma", "argmax"]:
        assert result[key].shape == (n,)

    assert result["lin"].shape == (101,)
    assert result["log_posterior"].shape == (n, 101)

    # Check true and inferred are correlated.
    pearsonr, pval = stats.pearsonr(result["gamma"], result["argmax"])
    assert pearsonr > 0.5
    assert pval < 0.1
