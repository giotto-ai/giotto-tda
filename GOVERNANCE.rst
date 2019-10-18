Contributing guidelines
=======================
Pull Request Checklist
----------------------

Before sending your pull requests, make sure you followed this list.
  - Read [contributing guidelines](GOVERNANCE.md).
  - Read [Code of Conduct](CODE_OF_CONDUCT.rst).
  - Ensure you have signed the [Contributor License Agreement (CLA)](DEED_OF_CONTRIBUTIONS.rst).
  - Check if my changes are consistent with the guidelines.
  - Changes are consistent with the Coding Style.
  - Run Unit Tests.
  
How to become a contributor and submit your own code
----------------------------------------------------
Contributor License Agreements
``````````````````````````````

We'd love to accept your patches! Before we can take them, we have to jump a couple of legal hurdles.
Please fill out either the individual or corporate Contributor License Agreement (CLA).
  - If you are an individual writing original source code and you're sure you own the intellectual property, then you'll need to sign an [individual CLA](DEED_OF_CONTRIBUTIONS.rst).
  - If you work for a company that wants to allow you to contribute your work, then you'll need to sign a corporate CLA (please write at <business@l2f.ch>).
  
Follow either of the two links above to access the appropriate CLA and instructions for how to sign and return it. Once we receive it, we'll be able to accept your pull requests.

**NOTE**: Only original source code from you and other people that have signed the CLA can be accepted into the main repository.

Contributing code
`````````````````
If you have improvements to Giotto, send us your pull requests! For thosejust getting started, Github has a[how to](https://help.github.com/articles/using-pull-requests/).
Giotto team members will be assigned to review your pull requests. Once thepull requests are approved and pass continuous integration checks, a Giotto team member will apply `ready to pull` label to your change. This means we areworking on getting your pull request submitted to our internal repository. Afterthe change has been submitted internally, your pull request will be mergedautomatically on GitHub.
If you want to contribute, start working through the Giotto codebase,navigate to the [Github "issues" tab](https://github.com/giotto-learn/giotto-learn/issues) and startlooking through interesting issues. These are issues that we believe are particularly well suited for outsidecontributions, often because we probably won't get to them right now. If youdecide to start on an issue, leave a comment so that other people know thatyou're working on it. If you want to help out, but not alone, use the issuecomment thread to coordinate.
Contribution guidelines and standards
`````````````````````````````````````
Before sending your pull request for review, make sure your changes are consistent with the guidelines and follow the Giotto coding style.

General guidelines and philosophy for contribution
``````````````````````````````````````````````````
  - Include unit tests when you contribute new features, as they help to a) prove that your code works correctly, and b) guard against future breaking changes to lower the maintenance cost. *Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient test coverage.*   
  - Keep API compatibility in mind when you change code in core Giotto. Reviewers of your pull request will comment on any API compatibility issues.*   When you contribute a new feature to Giotto, the maintenance burden is (by default) transferred to the Giotto team. This means that the benefit of the contribution must be compared against the cost of maintaining the    feature.
  - Full new features (e.g., a new op implementing a cutting-edge algorithm) typically will live in [giotto-learn/addons](https://github.com/giotto-learn/addons) to get some airtime before a decision is made regarding whether they are to be migrated to the core.
  
C++ coding style
''''''''''''''''
Changes to Giotto C/C++ code should conform to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
Use `clang-tidy` to check your C/C++ changes. To install `clang-tidy` on ubuntu:16.04, do:

``bashapt-get install -y clang-tidy``

You can check a C/C++ file by doing:

``bashclang-format <my_cc_file> --style=google > /tmp/my_cc_file.ccdiff <my_cc_file> /tmp/my_cc_file.cc``

Python coding style
'''''''''''''''''''
Changes to Giotto Python code should conform to [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)
Use `pylint` to check your Python changes. To install `pylint` and retrieve Giotto's custom style definition:

``bash pip install pylintwget -O /tmp/pylintrc https://raw.githubusercontent.com/giotto-learn/giotto-learn/master/giotto/tools/ci_build/pylintrc``

To check a file with `pylint`:

``bashpylint --rcfile=/tmp/pylintrc myfile.py``

Running unit tests
''''''''''''''''''
There are two ways to run Giotto unit tests.

1.  Using tools and libraries installed directly on your system. The election tool is pytest.
 
2.  Using [Azure](azure-pipelines.yml) and TensorFlow's CI scripts.  
