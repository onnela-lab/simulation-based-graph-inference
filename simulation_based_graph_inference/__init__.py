import torch as th


# Hack: do not save `_inv` which is a `weakref` that cannot be pickled (see
# https://github.com/pytorch/pytorch/pull/81707 for context and a fix).
def _Transform_getstate(self):
    state = self.__dict__.copy()
    state["_inv"] = None
    return state


th.distributions.Transform.__getstate__ = _Transform_getstate
