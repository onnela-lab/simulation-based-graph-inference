master_doc = 'README'
extensions = [
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
]
project = 'simulation_based_graph_inference'
napoleon_custom_sections = [('Returns', 'params_style')]
plot_formats = [
    ('png', 144),
]
plot_pre_code = """
from simulation_based_graph_inference.util import _plot_generated_graph
from simulation_based_graph_inference import generators
"""
html_theme = "nature"
add_module_names = False
