import argparse
import cmdstanpy
import networkx as nx
import numpy as np
import pathlib
import pickle
from scipy.spatial.distance import squareform
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
import typing

from ..config import GENERATOR_CONFIGURATIONS
from ..data import BatchedDataset


def infer_planted_partition_params(
        graph: nx.Graph, bias_loc: float, bias_scale: float, scale_conc: float, scale_rate: float,
        num_dims: int = 2, chains: int = 1, **kwargs) -> cmdstanpy.CmdStanMCMC:
    """
    Infer the parameters of the latent space model described by
    `Hoff et al. (2002) <https://doi.org/10.1198/016214502388618906>`__.

    Args:
        graph: Graph to use as data.
        num_dims: Number of embedding dimensions.

    Returns:
        result: Dictionary comprising the result.
    """
    adjacency = squareform(nx.to_numpy_array(graph)).astype(int)
    num_nodes = graph.number_of_nodes()
    data = {
        "num_nodes": num_nodes,
        "num_dims": num_dims,
        "adjacency": adjacency,
        # Prior parameters.
        "bias_loc": bias_loc,
        "bias_scale": bias_scale,
        "scale_conc": scale_conc,
        "scale_rate": scale_rate,
    }
    stan_file = pathlib.Path(__file__).parent / "infer_latent_space_params.stan"
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)
    return model.sample(data, chains, **kwargs)


def __main__(args: typing.Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", help="path to test set", required=True)
    parser.add_argument("--result", help="path at which to store evaluation on test set",
                        required=True)
    parser.add_argument("--iter_warmup", type=int, help="number of warmup steps")
    parser.add_argument("--iter_sampling", type=int, help="number of sampling steps", default=500)
    args = parser.parse_args(args)

    config = GENERATOR_CONFIGURATIONS["latent_space_graph"]

    dataset = BatchedDataset(args.test)
    result = {"args": vars(args)}
    for graph in tqdm(dataset):
        result.setdefault("params", {}).setdefault("bias", []).append(graph.bias.item())
        result.setdefault("params", {}).setdefault("scale", []).append(graph.scale.item())
        graph = to_networkx(graph).to_undirected()
        # Run the inference.
        fit = infer_planted_partition_params(
            graph, config.priors["bias"].loc.item(), config.priors["bias"].scale.item(),
            config.priors["scale"].concentration.item(), config.priors["scale"].rate.item(),
            config.generator_kwargs["num_dims"], iter_warmup=args.iter_warmup or args.iter_sampling,
            iter_sampling=args.iter_sampling,
        )
        # Save the samples for parameters.
        result.setdefault("samples", {}).setdefault("bias", []).append(fit.stan_variable("bias"))
        result.setdefault("samples", {}).setdefault("scale", []).append(fit.stan_variable("scale"))

    result["samples"] = {key: np.asarray(value) for key, value in result["samples"].items()}
    with open(args.result, "wb") as fp:
        pickle.dump(result, fp)


if __name__ == "__main__":  # pragma: no cover
    __main__()
