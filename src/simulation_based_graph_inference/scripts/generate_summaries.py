import pickle
import typing
from argparse import ArgumentParser
from pathlib import Path

import collectiontools
import networkx as nx
import numpy as np
from tqdm import tqdm

from .. import config
from ..data import BatchedDataset


def __main__(argv: typing.Optional[list[str]] = None) -> None:
    parser = ArgumentParser()
    parser.add_argument("dataset", help="Path to dataset.", type=Path)
    parser.add_argument("output", help="Path to output file.", type=Path)
    parser.add_argument(
        "--configuration",
        "-c",
        required=True,
        help="Configuration name.",
        choices=config.GENERATOR_CONFIGURATIONS,
    )
    args = parser.parse_args(argv)

    configuration = config.GENERATOR_CONFIGURATIONS[args.configuration]
    dataset = BatchedDataset(args.dataset)
    summaries = {}
    params = {name: [] for name in configuration.priors.keys()}
    for item in tqdm(dataset):
        # Load the graph as a networkx graph.
        graph = nx.Graph()
        num_nodes = item.num_nodes
        graph.add_nodes_from(range(num_nodes))
        graph.add_edges_from([(int(u), int(v)) for u, v in item.edge_index.T])
        assert graph.number_of_nodes() == num_nodes
        # Edge indices point in both directions so we have a factor of two.
        assert 2 * graph.number_of_edges() == item.edge_index.shape[1]

        # Evaluate summaries and mean/std-pool them.
        row = {
            "clustering": list(nx.clustering(graph).values()),  # pyright: ignore[reportAttributeAccessIssue]
            "degree": [degree for _, degree in nx.degree(graph)],
            "square_clustering": list(nx.square_clustering(graph).values()),  # pyright: ignore[reportAttributeAccessIssue]
            "effective_size": list(nx.effective_size(graph).values()),
            "average_neighbor_degree": list(nx.average_neighbor_degree(graph).values()),
        }
        for key, value in list(row.items()):
            del row[key]
            # Use nanmean/nanstd to handle NaN values (e.g., from isolated nodes).
            # Replace any remaining NaN (e.g., when all values are NaN) with 0.
            mean_val = np.nanmean(value) if value else 0.0
            std_val = np.nanstd(value) if value else 0.0
            row.update(
                {
                    f"{key}_mean": 0.0 if np.isnan(mean_val) else mean_val,
                    f"{key}_std": 0.0 if np.isnan(std_val) else std_val,
                }
            )

        collectiontools.append_values(summaries, row)

        # Extract parameter values.
        for name in params:
            params[name].append(item[name].item())

    summaries = collectiontools.map_values(np.asarray, summaries)

    # Verify no NaN or Inf values in summaries.
    for key, values in summaries.items():
        assert not np.isnan(values).any(), f"{key} contains NaN values"
        assert not np.isinf(values).any(), f"{key} contains Inf values"

    output = {
        "summaries": summaries,
        "params": {name: np.asarray(values) for name, values in params.items()},
        "configuration": args.configuration,
    }
    with open(args.output, "wb") as fp:
        pickle.dump(output, fp)


if __name__ == "__main__":
    __main__()
