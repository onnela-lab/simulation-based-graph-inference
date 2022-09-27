from setuptools import find_packages, setup


setup(
    name="simulation_based_graph_inference",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "cmdstanpy",
        "doit-interface>=0.1.6",
        "fasttr @ git+https://github.com/gstonge/fasttr",
        "matplotlib",
        "networkx",
        "numpy",
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
    ],
    extras_require={
        "tests": [
            "flake8",
            "pytest",
            "pytest-cov",
        ],
        "docs": [
            "sphinx",
        ]
    },
)
