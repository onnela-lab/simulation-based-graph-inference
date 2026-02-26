import numpy as np
import pytest
from simulation_based_graph_inference import config, data, models
import torch as th
from torch_geometric import nn as tgnn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


@pytest.fixture(params=config.GENERATOR_CONFIGURATIONS)
def generator_configuration(request):
    return request.param


@pytest.fixture
def batch(generator_configuration: str) -> Data:
    configuration = config.GENERATOR_CONFIGURATIONS[generator_configuration]
    dataset = data.SimulatedDataset(data.generate_data, (configuration, 12))
    loader = DataLoader(dataset, batch_size=32)  # type: ignore[arg-type]
    for batch in loader:
        return batch
    raise RuntimeError("No batch generated")


MODEL_CONFIGURATIONS = [
    {
        "conv": [
            tgnn.GINConv(lambda x: x),
            models.Normalize(tgnn.GINConv(lambda x: x)),
            models.Normalize(tgnn.GINConv(lambda x: x)),
        ],
        "dense": th.nn.Sequential(
            th.nn.Linear(3, 4),
            th.nn.Linear(4, 5),
        ),
    },
    {
        "conv": [
            tgnn.GINConv(lambda x: x),
            tgnn.GINConv(models.create_dense_nn([1, 4], th.nn.Tanh(), True)),
            tgnn.GINConv(lambda x: x),
        ],
        "dense": th.nn.Linear(9, 4),
    },
    {
        "conv": None,
        "dense": th.nn.Linear(1, 5),
    },
    {
        "conv": [
            models.Residual(
                tgnn.GINConv(models.create_dense_nn([1, 4], th.nn.Tanh(), True))
            ),
        ],
        "dense": th.nn.Linear(4, 5),
    },
]


@pytest.mark.parametrize("model_configuration", MODEL_CONFIGURATIONS)
def test_model_with_architectures(
    generator_configuration: str, batch, model_configuration: dict
):
    dists = config.GENERATOR_CONFIGURATIONS[generator_configuration].create_estimator()
    model = models.Model(
        model_configuration["conv"],
        model_configuration["dense"],
        dists,  # type: ignore[arg-type]
    )

    # Check that the features for the initial and transformed graph representation are on a sensible
    # scale. If they are not, training will fail almost immediately because we are in strange
    # regions of parameter space.
    graph_features = model.evaluate_graph_features(batch)
    assert (graph_features.abs().max() < 1e3).all()
    dense = model.dense(graph_features)
    assert (dense.abs().max() < 1e2).all()

    # Validate the outputs. This also verifies that the number of graph features is correct because
    # we do not use lazy modules in the dense network configuration above.
    dists, features = model(batch)
    for key, dist in dists.items():
        log_prob = dist.log_prob(batch[key])
        assert log_prob.shape == (batch.num_graphs,)
        assert log_prob.isfinite().all()


def test_normalize_mean_degree(batch):
    conv = [
        models.Normalize(tgnn.GINConv(lambda x: x), "mean_degree+1"),
    ]
    model = models.Model(conv, None, None)  # type: ignore[arg-type]
    graph_features = model.evaluate_graph_features(batch)
    th.testing.assert_close(graph_features, th.ones((batch.num_graphs, 1)))


@pytest.mark.parametrize("final_activation", [False, True])
def test_create_dense(final_activation: bool):
    dense = models.create_dense_nn([3, 4], th.nn.Tanh(), final_activation)
    x = dense(th.randn(10000, 3))
    if final_activation:
        assert (-1 <= x).all() and (x <= 1).all()


def test_distribution_module():
    parametrized = models.DistributionModule(
        th.distributions.Beta,
        concentration0=th.nn.Linear(1, 1),
        concentration1=th.nn.Linear(1, 1),
        transforms=[th.distributions.AffineTransform(3, -2)],
    )
    dist: th.distributions.Distribution = parametrized(th.ones(1))
    x = dist.sample([1000])
    assert x.min() > 1
    assert x.max() < 5


def test_residual_module(batch: Data) -> None:
    assert batch.num_nodes is not None
    residual = models.Residual(lambda x, edge_index: 0)
    x = th.randn(batch.num_nodes, 3)
    np.testing.assert_allclose(x, residual(x, batch.edge_index))

    residual = models.Residual(lambda x, edge_index: 2, method="scalar")
    residual._scalar = th.nn.Parameter(7 * th.ones([]))
    x = th.randn(batch.num_nodes, 3)
    np.testing.assert_allclose(2 + 7 * x, residual(x, batch.edge_index).detach())


def test_dense_residual_module() -> None:
    # Test identity residual
    residual = models.DenseResidual(lambda x: th.zeros_like(x))
    x = th.randn((10, 3))
    np.testing.assert_allclose(x, residual(x).detach())

    # Test scalar residual
    residual = models.DenseResidual(lambda x: 2 * th.ones_like(x), method="scalar")
    residual._scalar = th.nn.Parameter(7 * th.ones([]))
    x = th.randn((10, 3))
    np.testing.assert_allclose(2 + 7 * x, residual(x).detach())
