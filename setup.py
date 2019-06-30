from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='utility_cython',
    ext_modules=cythonize(
        Extension(
            "utility_cython",
            sources=["src/utility_cython.pyx"],
            include_dirs=[np.get_include()]
        )
    ),
    install_requires=["numpy"]
)
