import pytest
from simulation_based_graph_inference import config, data, models
import torch as th
import torch_geometric as tg


@pytest.fixture(params=config.GENERATOR_CONFIGURATIONS)
def generator_configuration(request):
    return request.param


@pytest.fixture
def batch(generator_configuration: str):
    generator, kwargs = config.GENERATOR_CONFIGURATIONS[generator_configuration]
    prior = config.get_prior(generator_configuration)
    dataset = data.SimulatedDataset(models.generate_data, (generator, 100, prior), kwargs)
    loader = tg.loader.DataLoader(dataset, batch_size=32)
    for batch in loader:
        return batch


MODEL_CONFIGURATIONS = [
    {
        "conv": [
            tg.nn.GINConv(lambda x: x),
            models.Normalize(tg.nn.GINConv(lambda x: x)),
            models.Normalize(tg.nn.GINConv(lambda x: x)),
        ],
        "dense": [
            th.nn.Linear(3, 4),
            th.nn.Linear(4, 5),
        ],
    },
    {
        "conv": [
            tg.nn.GINConv(lambda x: x),
            tg.nn.GINConv(models.create_dense_nn([1, 4], th.nn.Tanh(), True)),
            tg.nn.GINConv(lambda x: x),
        ],
        "dense": [
            th.nn.Linear(9, 4),
        ],
    },
    {
        "conv": None,
        "dense": th.nn.Linear(1, 5),
    }
]


@pytest.mark.parametrize("model_configuration", MODEL_CONFIGURATIONS)
def test_model_with_architectures(generator_configuration: str, batch, model_configuration: str):
    dists = config.get_parameterized_posterior_density_estimator(generator_configuration)
    model = models.Model(model_configuration["conv"], model_configuration["dense"], dists)

    # Check that the features for the initial and transformed graph representation are on a sensible
    # scale. If they are not, training will fail almost immediately because we are in strange
    # regions of parameter space.
    graph_features = model.evaluate_graph_features(batch)
    assert (graph_features.abs().max() < 1e3).all()
    dense = model.dense(graph_features)
    assert (dense.abs().max() < 1e2).all()

    # Validate the outputs. This also verifies that the number of graph features is correct because
    # we do not use lazy modules in the dense network configuration above.
    dists = model(batch)
    for key, dist in dists.items():
        log_prob = dist.log_prob(batch[key])
        assert log_prob.shape == (batch.num_graphs,)
        assert log_prob.isfinite().all()


def test_normalize_mean_degree(batch):
    conv = [
        models.Normalize(tg.nn.GINConv(lambda x: x), "mean_degree+1"),
    ]
    model = models.Model(conv, None, None)
    graph_features = model.evaluate_graph_features(batch)
    th.testing.assert_close(graph_features, th.ones((batch.num_graphs, 1)))


@pytest.mark.parametrize("final_activation", [False, True])
def test_create_dense(final_activation: bool):
    dense = models.create_dense_nn([3, 4], th.nn.Tanh(), final_activation)
    x = dense(th.randn(10000, 3))
    if final_activation:
        assert (-1 <= x).all() and (x <= 1).all()
