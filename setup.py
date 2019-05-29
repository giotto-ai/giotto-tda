from setuptools import setup

with open('README.rst') as f:
    long_description =  f.read()

hera_wasserstein = Extension(name                = "topological_learning.hera_wasserstein",
                             sources             = ["./dependencies/hera_wasserstein.pyx"],
                             language            = "c++",
                             extra_compile_args  = ["-std=c++14", "-I./dependencies/hera/geom_matching/wasserstein/include/"])

# gudhi_bottleneck = Extension(name                = "topological_learning.gudhi_bottleneck",
#                              sources             = ["./dependencies/gudhi_bottleneck.pyx"],
#                              language            = "c++",
#                              extra_compile_args  = ["-std=c++14", "-I./dependencies/gudhi/????"])

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
      install_requires=[
        # Common requirements
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'keras',
        # Gudhi
        'gudhi',
        # Ripser
        'Cython',
        'ripser',
        # Kepler Mapper
        'kmapper',
        # UMAP
        'umap-learn',
        # Synthetic datasets for TDA
        'tadasets'
      ],
      extras_require={
        'docs': [ # `pip install -e ".[docs]"``
            'sktda_docs_config',
            'pyyaml'
        ]
      },
      python_requires='>3.5',
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
