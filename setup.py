from setuptools import setup
from distutils.core import setup, Extension

with open('README.rst') as f:
    long_description = f.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

extensions = [
    Extension(name     = "giotto.external.bindings.hera_wasserstein",
              sources  = ["./giotto/external/bindings/hera_wasserstein.pyx"],
              language = "c++",
              extra_compile_args  = ["-std=c++14", "-I./giotto/external/hera/geom_matching/wasserstein/include/"])
]

try:
    from Cython.Distutils import build_ext
    from Cython.Build     import cythonize
    modules, cmds = cythonize(extensions), {"build_ext": build_ext}
    print("Cython found")

except ImportError:
    modules, cmds = [], {}
    print("Cython not found")


setup(name='giotto',
      version='0.0.1',
      description='giotto is a scikit-learn-based machine learning library that brings Topological Data Analysis to data scientists.',
      long_description=long_description,
      url='https://git.l2f.ch/g.tauzin/giotto.git',
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
      setup_requires=[
        'cython >= 0.29.7',
      ],
      install_requires=requirements,
      extras_require={
          'docs': [  # `pip install -e ".[docs]"``
              'sphinx',
              ]
      },
      test_suite='tests',
      tests_require=['pytest'],
      ext_modules=modules,
      zip_safe=False)
