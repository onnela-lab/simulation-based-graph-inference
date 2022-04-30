import os
import re
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
        define_macros=[('CYTHON_TRACE', '1')],
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

# Run find-replace on all generated cpp files (cf. https://github.com/cython/cython/issues/3929).
sources = {source for ext_module in ext_modules for source in ext_module.sources}
for source in sources:
    base, ext = os.path.splitext(source)
    assert not os.path.isabs(base), f"expected relative path but got {source}"
    assert ext == '.cpp', f"expected cpp extension but got {source}"
    relpath = base + ".pyx"
    abspath = os.path.join(os.getcwd(), relpath)
    with open(source) as fp:
        text = fp.read()
    text = re.sub(f"(?<!/){relpath}", abspath, text)
    with open(source, "w") as fp:
        fp.write(text)


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
    ext_modules=ext_modules,
)
