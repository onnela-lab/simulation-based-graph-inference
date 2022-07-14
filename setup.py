import os
from setuptools import find_packages, setup
from setuptools.extension import Extension
from Cython.Build import cythonize

define_macros = []
if os.environ.get('CYTHON_TRACE'):
    define_macros.append(('CYTHON_TRACE', '1'))

extensions = [
    Extension(
        name="*",
        sources=["simulation_based_graph_inference/**/*.pyx"],
        extra_compile_args=[
            "-std=c++17",
        ],
        language="c++",
        include_dirs=[
            "include",
        ],
        define_macros=define_macros,
    ),
]

ext_modules = cythonize(
    extensions,
    annotate=True,
    compiler_directives={
        'embedsignature': True,
        'binding': True,
        'language_level': 3,
        'linetrace': True,
    },
)

setup(
    name="simulation_based_graph_inference",
    packages=find_packages(),
    version="0.1.0",
    install_requires=[
        "cython",
        "doit-interface>=0.1.5",
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
    ext_modules=ext_modules,
)
