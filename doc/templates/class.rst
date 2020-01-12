:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

..
   Exclude sphinx-gallery generated examples since we use binder for now
   include:: {{module}}.{{objname}}.examples

.. raw:: html

    <div class="clearer"></div>
