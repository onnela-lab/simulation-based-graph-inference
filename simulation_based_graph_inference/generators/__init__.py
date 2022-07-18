from .duplication_divergence import duplication_complementation, duplication_mutation
from .spatial import geometric
from .random_attachment import poisson_random_attachment
from .redirection import redirection
from .web import web


__all__ = [
    "duplication_complementation",
    "duplication_mutation",
    "geometric",
    "poisson_random_attachment",
    "redirection",
    "web",
]
