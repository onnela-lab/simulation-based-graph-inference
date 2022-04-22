from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="*",
        sources=["simulation_based_graph_inference/*.pyx"],
        extra_compile_args=[
            "-std=c++17",
        ],
        language="c++",
        include_dirs=[
            "include",
        ],
    ),
]

setup(
    name="simulation_based_graph_inference",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "cython",
        "matplotlib",
        "networkx",
        "numpy",
        "torch",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
    ],
    setup_requires=[
        "cython",
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
    ext_modules=cythonize(
        extensions,
        annotate=True,
        compiler_directives={
            'embedsignature': True,
            'binding': True,
            'language_level': 3,
        },
    ),
)
