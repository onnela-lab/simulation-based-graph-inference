import os
import pytest
import tempfile
import torch as th


th.set_default_dtype(th.float64)


@pytest.fixture
def tmpwd():
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        yield tmp
        os.chdir(cwd)
