.. image:: https://www.giotto.ai/static/vector/logo.svg
   :width: 850

|Azure|_ |Azure-cov|_ |Azure-test|_ |binder|_

.. |Azure| image:: https://dev.azure.com/maintainers/Giotto/_apis/build/status/giotto-ai.giotto-learn?branchName=master
.. _Azure: https://dev.azure.com/maintainers/Giotto/_build/latest?definitionId=2&branchName=master

.. |Azure-cov| image:: https://img.shields.io/badge/Coverage-93%25-passed
.. _Azure-cov: https://dev.azure.com/maintainers/Giotto/_build/results?buildId=6&view=codecoverage-tab

.. |Azure-test| image:: https://img.shields.io/badge/Testing-Passed-brightgreen
.. _Azure-test: https://dev.azure.com/maintainers/Giotto/_build/results?buildId=6&view=ms.vss-test-web.build-test-results-tab

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/giotto-ai/giotto-learn/master?filepath=examples


giotto-learn
============


giotto-learn is a high performance topological machine learning toolbox in Python built on top of
scikit-learn and is distributed under the Apache 2.0 license. It is part of the Giotto open-source project.

Website: https://giotto.ai


Installation
------------

Dependencies
~~~~~~~~~~~~

giotto-learn requires:

- Python (>= 3.5)
- scikit-learn (>= 0.21.3)
- NumPy (>= 1.11.0)
- SciPy (>= 0.17.0)
- joblib (>= 0.11)

For running the examples jupyter, matplotlib and plotly are required.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of numpy and scipy,
the easiest way to install giotto-learn is using ``pip``   ::

    pip install -U giotto-learn

Documentation
-------------

- HTML documentation (stable release): https://docs.giotto.ai

Contributing
------------

We welcome new contributors of all experience levels. The Giotto
community goals are to be helpful, welcoming, and effective. To learn more about
making a contribution to giotto-learn, please see the `CONTRIBUTING.rst
<https://github.com/giotto-ai/giotto-learn/blob/master/CONTRIBUTING.rst>`_ file.

Developer installation
~~~~~~~~~~~~~~~~~~~~~~~

C++ dependencies:
'''''''''''''''''

-  C++14 compatible compiler
-  CMake >= 3.9
-  Boost >= 1.56

Source code
'''''''''''

You can check the latest sources with the command::

    git clone https://github.com/giotto-ai/giotto-learn.git


To install:
'''''''''''

.. code-block:: bash

   cd giotto-learn
   pip install -e .

From there any change in the library files will be immediately available on your machine.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the
source directory::

    pytest giotto


Changelog
---------

See the `RELEASE.rst <https://github.com/giotto-ai/giotto-learn/blob/master/RELEASE.rst>`__ file
for a history of notable changes to giotto-learn.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/giotto-ai/giotto-learn
- Download releases: https://pypi.org/project/giotto-learn/
- Issue tracker: https://github.com/giotto-ai/giotto-learn/issues


Contacts:
---------

maintainers@giotto.ai
