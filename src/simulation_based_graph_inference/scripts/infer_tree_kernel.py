import argparse
from fasttr import HistorySampler
import networkx as nx
import numpy as np
import pickle
from scipy import integrate, optimize, special
import torch as th
from torch.distributions import constraints
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
from ..data import BatchedDataset
from ..config import GENERATOR_CONFIGURATIONS


class TreeKernelPosterior:
    """
    Posterior for the kernel of a randomly grown graph (see
    https://doi.org/10.1103/PhysRevLett.126.038301 for details).

    Args:
        graph: Graph from which to infer kernel parameters.
        num_history_samples: Number of histories to infer.
        lim: Half width of the interval around the MAP estimate to consider for obtaining the
            normalization constant.
    """

    def __init__(
        self,
        graph: nx.Graph,
        prior: th.distributions.Distribution,
        num_history_samples: int = 100,
    ):
        self.graph = graph
        self.sampler = HistorySampler(graph)
        self.sampler.sample(num_history_samples)
        self.prior = prior
        self.support: constraints.interval = self.prior.support

        # Find the maximum so we get better numerics.
        self.max = 0
        result = optimize.minimize_scalar(
            lambda gamma: -self.log_target(gamma),
            method="bounded",
            bounds=[self.support.lower_bound, self.support.upper_bound],
        )
        assert result.success
        self.argmax = result.x

        # Then set the maximum so we can subtract it from the target. This means
        # that evaluating the target at argmax should be zero.
        self.max = self.log_target(self.argmax)
        assert abs(self.log_target(self.argmax)) < 1e-9

        # Integrate to find the normalization constant.
        self.log_norm = 0
        norm, *_ = integrate.quad(
            lambda x: np.exp(self.log_prob(x)),
            self.support.lower_bound,
            self.support.upper_bound,
        )
        self.log_norm = np.log(norm)

    def log_target(self, gamma):
        """
        Evaluate the unnormalized log posterior.
        """
        self.sampler.set_kernel(kernel=lambda k: k**gamma)
        log_likelihoods = self.sampler.get_log_posterior()
        log_posterior = (
            special.logsumexp(log_likelihoods)
            - np.log(len(log_likelihoods))
            - self.max
            + self.prior.log_prob(th.as_tensor(gamma)).item()
        )
        return log_posterior

    def log_prob(self, gamma):
        """
        Evaluate the log posterior.
        """
        if isinstance(gamma, float):
            return self.log_target(gamma) - self.log_norm
        return np.reshape([self.log_prob(x) for x in np.ravel(gamma)], np.shape(gamma))


def __main__(args: list[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="path to test set", required=True)
    parser.add_argument(
        "--result", help="path at which to store evaluation on test set", required=True
    )
    args = parser.parse_args(args)

    prior = GENERATOR_CONFIGURATIONS["gn_graph"].priors["gamma"]
    # Use 101 elements just to catch accidental issues with tensor shapes.
    lin = np.linspace(prior.support.lower_bound, prior.support.upper_bound, 101)

    dataset = BatchedDataset(args.test)
    result = {}
    for graph in tqdm(dataset):
        gamma = graph.gamma.item()
        graph = to_networkx(graph).to_undirected()
        posterior = TreeKernelPosterior(graph, prior)
        log_posterior = posterior.log_prob(lin)

        result.setdefault("gamma", []).append(gamma)
        result.setdefault("argmax", []).append(posterior.argmax)
        result.setdefault("log_prob", []).append(posterior.log_prob(gamma))
        result.setdefault("log_posterior", []).append(log_posterior)

    result = {key: np.asarray(value) for key, value in result.items()}
    result["args"] = vars(args)
    result["lin"] = lin

    with open(args.result, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":  # pragma: no cover
    __main__()
