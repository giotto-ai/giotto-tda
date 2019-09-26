#! /usr/bin/env python
"""Toolbox for Topological Data Analysis in machine learning."""

import os
import codecs
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.test import test as TestCommand
from shutil import copyfile, copymode

version_file = os.path.join('giotto', '_version.py')
with open(version_file) as f:
    exec(f.read())

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

DISTNAME = 'giotto-learn'
DESCRIPTION = 'Toolbox for Topological Data Analysis dataset in machine learning.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = 'Matteo Caorsi'
MAINTAINER_EMAIL = 'm.caorsi@l2f.ch'
URL = 'https://github.com/matteocao/giotto-learn/'
LICENSE = 'Apache 2.0'
DOWNLOAD_URL = 'https://github.com/matteocao/giotto-learn/'
VERSION = __version__
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: C',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
KEYWORDS = 'machine learning topological data analysis persistent ' + \
    'homology, persistence diagrams'
INSTALL_REQUIRES = requirements
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
        'pytest-benchmark'],
    'binary': [
        'wheel',
        'twine'],
    'doc': [
        'sphinx',
        'sphinx-gallery',
        'sphinx-issues',
        'sphinx_rtd_theme',
        'numpydoc'],
    'examples': [
        'jupyter',
        'matplotlib',
        'plotly',
        'pandas',
        'keras',
        'keras-metrics',
        'tensorflow']
}


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      keywords=KEYWORDS,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      ext_modules=[CMakeExtension('giotto')],
      cmdclass=dict(build_ext=CMakeBuild))
