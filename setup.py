from setuptools import setup


with open('README.rst') as f:
    long_description =  f.read()

setup(name='topological_ml',
      version='0.1',
      description='This package structures and makes accessible to all the tools used or developed by the research team to do Topological Data Analysis within a sk-learn+Keras Machine Learning',
      long_description=long_description,
      url='https://git.l2f.ch/g.tauzin/topological_ml.git',
      author='Guillaume Tauzin',
      author_email='g.tauzin@l2f.ch',
      license='MIT',
      packages=['topological_ml'],
      include_package_data=True,
      keywords='topology data analysis, persistent homology, persistence diagrams, uniform manifold approximation and projection',
      install_requires=[
        # Common requirements
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
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
