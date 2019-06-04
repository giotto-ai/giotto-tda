from setuptools import setup
from distutils.core import setup, Extension
from Cython.Build import cythonize

with open('README.rst') as f:
    long_description =  f.read()

hera_wasserstein = Extension(name                = "topological_learning.hera_wasserstein",
                             sources             = ["./topological_learning/dependencies/hera_wasserstein.pyx"],
                             language            = "c++")

gudhi_bottleneck = Extension(name                = "topological_learning.gudhi_bottleneck",
                             sources             = ["./topological_learning/dependencies/gudhi_bottleneck.pyx"],
                             language            = "c++")

try:
    from Cython.Distutils import build_ext
    from Cython.Build     import cythonize
    modules, cmds = cythonize([hera_wasserstein]), {"build_ext": build_ext}
    print("Cython found")

except ImportError:
    modules, cmds = [], {}
    print("Cython not found")

setup(name='topological_learning',
      version='0.1',
      description='This package structures and makes accessible to all the tools used or developed by the research team to do Topological Data Analysis within a sk-learn+Keras Machine Learning',
      long_description=long_description,
      url='https://git.l2f.ch/g.tauzin/topological_learning.git',
      author='Guillaume Tauzin',
      author_email='g.tauzin@l2f.ch',
      license='MIT',
      packages=['topological_learning'],
      include_package_data=True,
      keywords='topology data analysis, persistent homology, persistence diagrams, uniform manifold approximation and projection',
      python_requires='>=3.5',
      install_requires=[
        # Common requirements
        'numpy',
        'pandas',
        'pyarrow',
        #'fastparquet',
        #'python-snappy',
        'scipy',
        'scikit-learn',
        'keras',
        'tensorflow',
        'umap-learn',
        'numba',
        # Gudhi
        'gudhi',
        # Ripser
        'ripser',
      ],
      extras_require={
        'docs': [ # `pip install -e ".[docs]"``
            'sktda_docs_config',
            'pyyaml'
        ]
      },
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
