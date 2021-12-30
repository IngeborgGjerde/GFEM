.. Generalized Finite Element Methods in FEniCS documentation master file, created by
   sphinx-quickstart on Fri Dec 17 15:54:00 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GFEniCS's documentation!
========================================================================

In this module we try to implement Generalized Finite Element Methods  (GFEM) in FEniCS.

The GFE method requires two things:
   #. A partition of unity (PUM) and local approximation space
   #. Suitable quadrature rules

As partition of unity we use the hat functions and for the local approximating space we use analytic solutions. 

Quadrature rules are difficult to override in fenics so we use a mapping between our original mesh and a refined one. 

You can test and compare the different FEMs using gfem_example.py

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   gfem_ex.rst   
   fems.rst
   quad.rst
   testproblems.rst




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
