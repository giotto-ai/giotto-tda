from setuptools import setup
from distutils.core import setup, Extension

with open('README.rst') as f:
    long_description =  f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

hera_wasserstein = Extension(name     = "topological_learning.hera_wasserstein",
                             sources  = ["./topological_learning/dependencies/hera_wasserstein.pyx"],
                             language = "c++")

gudhi_bottleneck = Extension(name     = "topological_learning.gudhi_bottleneck",
                             sources  = ["./topological_learning/dependencies/gudhi_bottleneck.pyx"],
                             language = "c++")

try:
    from Cython.Distutils import build_ext
    from Cython.Build     import cythonize
    modules, cmds = cythonize([hera_wasserstein]), {"build_ext": build_ext}
    print("Cython found")

except ImportError:
    modules, cmds = [], {}
    print("Cython not found")

setup(name='topological_learning',
      version='0.0.1',
      description='This package structures and makes accessible to all the tools used or developed by the research team to do Topological Data Analysis within a sk-learn+Keras Machine Learning framework.',
      long_description=long_description,
      url='https://git.l2f.ch/g.tauzin/topological_learning.git',
      project_urls={
        "Issue Tracker": "https://git.l2f.ch/topological_learning/issues",
        "Documentation": "https://git.l2f.ch/topological_learning/",
        "Source Code": "https://git.l2f.ch/topological_learning/tree/master"
      },
      author='Guillaume Tauzin',
      author_email='g.tauzin@l2f.ch',
      license='MIT',
      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
      ],
      packages=['topological_learning'],
      include_package_data=True,
      keywords='topological data analysis, persistent homology, persistence diagrams',
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
