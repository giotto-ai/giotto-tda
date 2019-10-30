Contributing guidelines
=======================

Pull Request Checklist
----------------------

Before sending your pull requests, make sure you followed this list.
  - Read the `contributing guidelines <https://github.com/giotto-learn/giotto-learn/blob/master/GOVERNANCE.rst>`_.
  - Read the `code of conduct <https://github.com/giotto-learn/giotto-learn/blob/master/CODE_OF_CONDUCT.rst>`_.
  - Ensure you have signed the `contributor license agreement (CLA) <https://cla-assistant.io/giotto-learn/giotto-learn>`_.
  - Check if the changes are consistent with the guidelines.
  - Changes are consistent with the Coding Style.
  - Run Unit Tests.

How to become a contributor and submit your own code
----------------------------------------------------

Contributor License Agreements
------------------------------

In order to become a contributor of Giotto, the first step is to sign the
`contributor license agreement (CLA) <https://cla-assistant.io/giotto-learn/giotto-learn>`_.
**NOTE**: Only original source code from you and other people that have signed
the CLA can be accepted into the main repository.

Contributing code
-----------------

If you have improvements to Giotto, do not hesitate to send us pull requests!
Please follow the Github how to (https://help.github.com/articles/using-pull-requests/).
The Giotto Team will review your pull requests. Once the pull requests are approved and pass continuous integration checks, the
Giotto team will work on getting your pull request submitted to our GitHub
repository. Eventually, your pull request will be merged automatically on GitHub.
If you want to contribute, start working through the Giotto codebase,
navigate to the `GitHub issue tab <https://github.com/giotto-learn/giotto-learn/issues`_
and start looking through interesting issues. These are issues that we believe
are particularly well suited for outside contributions, often because we
probably won't get to them right now. If you decide to start on an issue, leave
a comment so that other people know that you're working on it. If you want to
help out, but not alone, use the issue comment thread to coordinate.

Contribution guidelines and standards
-------------------------------------

Before sending your pull request for review, make sure your changes are
consistent with the guidelines and follow the coding style below.

General guidelines and philosophy for contribution
--------------------------------------------------

* Include unit tests when you contribute new features, as they help to
  a) prove that your code works correctly, and
  b) guard against future breaking changes to lower the maintenance cost.
* Bug fixes also generally require unit tests, because the presence of bugs
  usually indicates insufficient test coverage.
* Keep API compatibility in mind when you change code in core Giotto.
* Clearly define your exceptions using the utils functions and test the exceptions.
* When you contribute a new feature to Giotto, the maintenance burden is   
  (by default) transferred to the Giotto team. This means that the benefit   
  of the contribution must be compared against the cost of maintaining the   
  feature.

C++ coding style
----------------

Changes to Giotto C/C++ code should conform to `Google C++ Style Guide <https://google.github.io/styleguide/cppguide.html>`_.
Use `clang-tidy` to check your C/C++ changes. To install `clang-tidy` on
ubuntu:16.04, do:


.. code-block:: bash

    apt-get install -y clang-tidy

You can check a C/C++ file by doing:

.. code-block:: bash

    clang-format <my_cc_file> --style=google > /tmp/my_cc_file.ccdiff <my_cc_file> /tmp/my_cc_file.cc

Python coding style
-------------------

Changes to Giotto Python code should conform to PEP8 directives.
Use `flake8` to check your Python changes. To install `flake8` just do

.. code-block:: python

    pip install flake8

You can use `flake8` on your python code via the following instructions:

.. code-block:: python

    flake8 name_of_your_script.py

Running unit tests
------------------

There are two ways to run Giotto unit tests.

1. Using tools and libraries installed directly on your system. The election tool is `pytest`. To install `pytest` just do

.. code-block:: python

    pip install pytest

You can use `pytest` on your python code via the following instructions:

.. code-block:: python

    pytest name_of_your_script.py

2. Using Azure (azure-pipelines.yml) and Giotto's CI scripts.
