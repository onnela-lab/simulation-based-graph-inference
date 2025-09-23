master_doc = "README"
extensions = [
    "matplotlib.sphinxext.plot_directive",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
]
project = "simulation_based_graph_inference"
napoleon_custom_sections = [("Returns", "params_style")]
plot_formats = [
    ("png", 144),
]
plot_pre_code = """
from simulation_based_graph_inference.util import _plot_generated_graph
from simulation_based_graph_inference import generators
"""
html_theme = "nature"
add_module_names = False
autodoc_typehints_format = "short"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
exclude_patterns = ["playground", ".venv"]
