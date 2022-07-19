from .duplication_divergence import duplication_complementation_graph, duplication_mutation_graph
from .attachment import random_attachment_graph, degree_attachment_graph, rank_attachment_graph
from .redirection import redirection_graph
from .web import web_graph


__all__ = [
    "duplication_complementation_graph",
    "duplication_mutation_graph",
    "random_attachment_graph",
    "degree_attachment_graph",
    "rank_attachment_graph",
    "redirection_graph",
    "web_graph",
]
