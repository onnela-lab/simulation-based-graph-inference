from doit_interface import dict2args
import os
import pickle
from simulation_based_graph_inference.scripts import generate_data, infer_latent_space_params


def test_infer_latent_space_params(tmpwd: str):
    n = 2
    iter_sampling = 3
    datadir = os.path.join(tmpwd, "data")
    generate_data.__main__(dict2args(configuration="latent_space_graph", batch_size=n,
                                     num_batches=1, directory=datadir))
    filename = os.path.join(tmpwd, "result.pkl")
    infer_latent_space_params.__main__(dict2args(test=datadir, result=filename,
                                                 iter_sampling=iter_sampling))

    with open(filename, "rb") as fp:
        result = pickle.load(fp)
    for key in ["bias", "scale"]:
        assert result["samples"][key].shape == (n, iter_sampling)
