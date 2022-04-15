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
