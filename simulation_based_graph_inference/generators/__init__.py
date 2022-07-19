from .duplication_divergence import duplication_complementation_graph, duplication_mutation_graph
from .random_attachment import poisson_random_attachment_graph
from .redirection import redirection_graph
from .web import web_graph


__all__ = [
    "duplication_complementation_graph",
    "duplication_mutation_graph",
    "poisson_random_attachment_graph",
    "redirection_graph",
    "web_graph",
]
