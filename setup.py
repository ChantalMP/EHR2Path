from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "convert_column",  # Name of the generated extension
        ["convert_column.pyx"],  # Your Cython source file
        include_dirs=[np.get_include()],  # Include NumPy headers
        extra_compile_args=["-O3"],  # Optimization flag
        extra_link_args=["-O3"],  # Ensure -O3 is used during linking as well
    )
]

setup(
    ext_modules=cythonize(extensions),
)
