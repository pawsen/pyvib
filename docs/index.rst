.. currentmodule:: pyvib

Welcome to the pyvib documentation
==================================

Pyvib is a python program for analyzing nonlinear vibrations and estimating(and
simulating) models from measured data.

The highlights are

- Analyzing (working on data):
   **Restoring force surface** (RFS) to visualize the functional form of the
   nonlinearties.

   **Wavelet transform** gives a frequency and time resolution plots from where
   the type of nonlinearity can be deducted.

   **Quantification of noise and nonlinear distortion** using the best linear
   approximation (BLA).
* Modeling (working on data):
   White box, using subspace identification and specified polynomials and
   splines to model the identified nonlinearities, known as **frequency
   nonlinear system identification** (FNSI).

   Black box, using **polynomial nonlinear state-space** (PNLSS).
* Understanding (working on identified FE model):
   **Harmonic balance continuation** to reveal bifurcations and jumps

   **Nonlinear normal modes** to reveal internal resonances and energy transfer
   between modes [#nnm]_.

See the `references`_ for description of the methods and the `credits`_.

Usage
-----

The :doc:`user-guide` provides a detailed description of the `pnlss example
<https://github.com/pawsen/pyvib/tree/master/examples/pnlss/tutorial.py>`_, to
show how pyvib works. See the `examples directory
<https://github.com/pawsen/pyvib/tree/master/examples>`_ for additional examples.

The :doc:`modules/index` documentation provides API-level documentation.
:doc:`contributing` shows how to contribute to the program or report bugs.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   user-guide
   modules/index
   contributing


Credits
-------

The PNLSS functionality is a translation of a matlab program written by the
staff at `Vrije Universiteit Brussel
<http://homepages.vub.ac.be/~ktiels/pnlss.html>`_ (VUB) . The documentation is
written by `Koen Tiels <https://katalog.uu.se/profile/?id=N18-309>`_ and he have
also written a short primer on `pnlss <http://homepages.vub.ac.be/~ktiels/pnlss_v1_0.pdf>`_

The FNSI method is developed by `Jean-Philippe Noël <http://www.jpnoel.be>`_. He
also kindly provided a matlab implementation of the spline method during my
thesis.

References
----------

PNLSS: Identification of nonlinear systems using polynomial nonlinear state
space models.
`PhD thesis
<http://homepages.ulb.ac.be/~jschouk/johan/phdthesis/16_phdjpaduart.pdf>`_.
`article <https://sci-hub.tw/10.1016/j.automatica.2010.01.001>`_

FNSI: Frequency-domain subspace identification for nonlinear mechanical systems.
PhD thesis Not longer available online. I will ask if it can be uploaded.
`article <https://sci-hub.tw/10.1016/j.ymssp.2013.06.034>`_

RFS and wavelet transform are described in the PhD thesis of J.P. Noël.

HBC: The harmonic balance method for bifurcation analysis of large-scale
nonlinear mechanical systems.
PhD thesis Not longer available online.
`article <https://sci-hub.tw/10.1016/j.cma.2015.07.017>`_

NNM: Nonlinear normal modes, Part II: Toward a practical computation using
numerical continuation techniques
PhD thesis by Maxime Peeters no longer available.
`article <https://sci-hub.tw/10.1016/j.ymssp.2008.04.003>`_

You can also read my master thesis which describe the methods and provides
additional references.

The project is on `GitHub`_ and you are welcome to modify, comment or suggest changes to the code.

Written by `Paw <pawsen@gmail.com>`_, PhD student at the University of
Liege.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _pawsen@gmail.com: mailto:pawsen@gmail.com
.. _GitHub: https://github.com/pawsen/pyvib

.. rubric:: Footnotes
.. [#nnm]  Note this uses the shooting method. For a usable, practical implementation the harmonic balance method should be used instead.
