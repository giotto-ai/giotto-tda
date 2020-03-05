
===
FAQ
===

How to cite?
------------

Cite the paper!

.. code-block:: latex

   @inproceedings{whatever_2014,
   	address = {Moon},
   	series = {{ABC}'14},
   	title = {A fancy paper},
   	isbn = {978-1-4503-2594-3},
   	url = {https://doi.org/10.123/abs},
   	doi = {123.12345},
   	booktitle = {A fancy book},
   	publisher = {Association for Computing Machinery},
   	author = {Author, Well-Known},
   	month = jun,
   	year = {2014},
   	keywords = {giotto},
   	pages = {-1-4}
   }


I am a researcher. Can I use the giotto-tda in my project?
----------------------------------------------------------

Of course, the license is very permissive. For more information, please contact the L2F team at
business@l2f.ch.

I cannot install `giotto-tda` on Windows.
-----------------------------------------

We are trying our best to support a variety of most-used operating systems.
If you experience any trouble, it is likely that others already have and reported it.
Please consult the list of `issues <https://github.com/giotto-ai/giotto-tda/issues?q=is%3Aissue>`,
including the closed ones, and open a new one in case you did not find help.

There are many TDA-libraries available. How is `giotto-tda` different?
----------------------------------------------------------------------

Giotto-tda is oriented towards machine learning (for details, see the :ref:`guiding principles <guiding_principles>`).
This philosophy is in contrast with other reference librairies, like `GUDHI <https://gudhi.inria.fr/doc/latest/index.html>`_,
which provide more functionality, at the expense of being less adapted to, f.ex. batch processing, or having no unified API.