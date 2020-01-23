#! /usr/bin/env python
"""Toolbox for Machine Learning using Topological Data Analysis."""

import os
import codecs
import re
import sys
import platform
import subprocess

from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


version_file = os.path.join('gtda', '_version.py')
with open(version_file) as f:
    exec(f.read())

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

DISTNAME = 'giotto-tda'
DESCRIPTION = 'Toolbox for Machine Learning using Topological Data Analysis.'
with codecs.open('README.rst', encoding='utf-8-sig') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_TYPE = 'text/x-rst'
MAINTAINER = 'Umberto Lupo, Lewis Tunstall'
MAINTAINER_EMAIL = 'maintainers@giotto.ai'
URL = 'https://github.com/giotto-ai/giotto-tda'
LICENSE = 'GNU AGPLv3'
DOWNLOAD_URL = 'https://github.com/giotto-ai/giotto-tda/tarball/v0.1.4'
VERSION = __version__ # noqa
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: C++',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']
KEYWORDS = 'machine learning, topological data analysis, persistent ' + \
    'homology, persistence diagrams, Mapper'
INSTALL_REQUIRES = requirements
is_system_win = platform.system() == 'Windows'
if is_system_win:
    python_ver = sys.version_info
    python_ver_1 = str(python_ver.major) + str(python_ver.minor)
    if python_ver_1 == '38':
        python_ver_2 = python_ver_1
    else:
        python_ver_2 = python_ver_1 + 'm'
    pycairo_whl_url = \
        'https://storage.googleapis.com/l2f-open-models/giotto' \
        '-learn/windows-binaries/pycairo/pycairo-1.18.2-cp{}' \
        '-cp{}-win_amd64.whl'.format(python_ver_1, python_ver_2)
    igraph_whl_url = \
        'https://storage.googleapis.com/l2f-open-models/giotto' \
        '-learn/windows-binaries/python-igraph/python_igraph-' \
        '0.7.1.post6-cp{}-cp{}-win_amd64.whl'.\
        format(python_ver_1, python_ver_2)
    INSTALL_REQUIRES.append('pycairo @ {}'.format(pycairo_whl_url))
    INSTALL_REQUIRES.append('python-igraph @ {}'.format(igraph_whl_url))
else:
    INSTALL_REQUIRES.append('python-igraph')
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov',
        'pytest-azurepipelines',
        'pytest-benchmark',
        'jupyter_contrib_nbextensions',
        'flake8',
        'hypothesis'],
    'doc': [
        'sphinx',
        'sphinx-gallery',
        'sphinx-issues',
        'sphinx_rtd_theme',
        'numpydoc'],
    'examples': [
        'jupyter',
        'matplotlib',
        'plotly']
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
            raise RuntimeError("CMake must be installed to build the "
                               " following extensions: " +
                               " , ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)',
                                                   out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        self.install_dependencies()

        for ext in self.extensions:
            self.build_extension(ext)

    def install_dependencies(self):
        dir_start = os.getcwd()
        dir_pybind11 = os.path.join(dir_start,
                                    'gtda', 'externals', 'pybind11')
        if os.path.exists(dir_pybind11):
            return 0
        os.mkdir(dir_pybind11)
        subprocess.check_call(['git', 'clone',
                               'https://github.com/pybind/pybind11.git',
                               dir_pybind11])
        os.chdir(dir_pybind11)
        dir_build = os.path.join(dir_pybind11, 'build')
        os.mkdir(dir_build)
        os.chdir(dir_build)
        cmake_cmd1 = ['cmake', '-DPYBIND11_TEST=OFF', '..']
        if platform.system() == "Windows":
            cmake_cmd2 = ['cmake', '--install', '.']
            if sys.maxsize > 2**32:
                cmake_cmd1 += ['-A', 'x64']
        else:
            cmake_cmd2 = ['make', 'install']
            cmake_cmd2_sudo = ['sudo', 'make', 'install']
        subprocess.check_call(cmake_cmd1, cwd=dir_build)
        try:
            subprocess.check_call(cmake_cmd2, cwd=dir_build)
        except:  # noqa
            subprocess.check_call(cmake_cmd2_sudo, cwd=dir_build)
        os.chdir(dir_start)

        subprocess.check_call(['git', 'submodule', 'update',
                               '--init', '--recursive'])

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args,
                              cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                              cwd=self.build_temp)


setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      keywords=KEYWORDS,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      ext_modules=[CMakeExtension('gtda')],
      cmdclass=dict(build_ext=CMakeBuild))
