.. highlight:: sh
.. currentmodule:: pyvib

Installation
============

This tutorial will walk you through the process of installing pyvib. To follow,
you really only need two basic things:

* A working `Python3 <http://www.python.org>`_ installation. Python3.6 is
  required.
* The python packages *numpy*, *scipy*, *matplotlib*

Step 0: Install prerequisites
-----------------------------
**Recommended**:

The necessary Python packages can be installed via the Anaconda
Python distribution (https://www.anaconda.com/download/). Python 3 is needed.
Anaconda is a collection of often used python packages, the python program and
`conda`, a open source package management system. It runs on Windows, macOS and
Linux. Especially on windows it makes the installation process simpler, as numpy
and scipy depends on external libraries which can be difficult to install, but
are automatically included when `conda` is used to install the python packages.

If you do not want to install all the packages included in Anaconda (some 3GB of
space), `miniconda <https://conda.io/miniconda.html>`_ can be installed instead
which just includes conda and the python program. The needed packages NumPy,
SciPy and matplotlib can be installed in miniconda via the command::

    $ conda install numpy scipy matplotlib

**Manual**

If you do not want to use `conda`, the packages can be installed on a
debian-like(linux) system::

    $ sudo apt install python3 python3-pip
    $ pip3 install numpy scipy matplotlib

On windows, check these wheels (binaries) for pip:
`numpy <https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy>`_
`scipy <https://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy>`_

Step 1: Download and unpack pyvib
-----------------------------------

`Download pyvib <https://github.com/pawsen/pyvib/archive/master.zip>`_ and unpack it::

    $ unzip pyvib-master.zip

If you're downloading from git, do::

    $ git clone https://github.com/pawsen/pyvib.git

Step 2: Build pyvib
--------------------

Just type::

    $ cd pyvib-master # if you're not there already
    $ python setup.py install

Step 3: Test pyvib
-------------------
Just type::

    $ cd test
    $ py.test

Advanced
--------

Get better speed by linking numpy and scipy with optimised blas libraries. This
is taken care of if you uses the anaconda distribution. Anaconda default ships
with `Intel Math Kernel Library` (MKL) blas, which is probably the fastest blas
implementation. It is Open License (not Open Source). If you wish an Open Source
blas, numpy/scipy linked to OpenBlas can be installed with::

    $ conda install -c conda-forge numpy scipy

`-c` specifies the channel to install from and `conda-forge` is a community
driven package central. In the folder `~/miniconda3/conda-meta/` you can see
which blas the current numpy is shipped with.

Another reason the prefer OpenBlas is space considerations. MKL takes around
800MB whereas OpenBlas is < 10MB.

If you install numpy/scipy from pip they are linked towards OpenBlas.
