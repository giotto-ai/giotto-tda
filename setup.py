from setuptools import setup

with open('README.rst') as f:
    long_description =  f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='giotto',
      version='0.0.1',
      description='giotto is a scikit-learn-based machine learning library that brings Topological Data Analysis to data scientists.',
      long_description=long_description,
      url='https://git.l2f.ch/g.tauzin/topological_learning.git',
      project_urls={
          "Issue Tracker": "TBA",
          "Documentation": "TBA",
          "Source Code": "TBA"
      },
      maintainer='Guillaume Tauzin',
      maintainer_email='guillaume.tauzin@epfl.ch',
      license='TBA',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: TBA',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
      ],
      packages=['giotto'],
      include_package_data=True,
      keywords='machine learning, topological data analysis, persistent homology, persistence diagrams',
      python_requires='>=3.5',
      install_requires=requirements,
      extras_require={
        'docs': [ # `pip install -e ".[docs]"``
            'sphinx',
        ]
      },
      test_suite='tests',
      tests_require=['pytest'],
      zip_safe=False)
