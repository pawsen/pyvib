.. currentmodule:: pyvib

User Guide
==========

Below is a detailed description of the `pnlss example
<https://github.com/pawsen/pyvib/tree/master/examples/pnlss/tutorial.py>`_, to
show how pyvib works. See the
`examples directory <https://github.com/pawsen/pyvib/tree/master/examples>`_ for
additional examples and :doc:`modules/index` for description of individual functions.

For a description of the pnlss methods, see the `guide from VUB
<http://homepages.vub.ac.be/~ktiels/pnlss_v1_0.pdf>`_. They also provide a
`matlab program <http://homepages.vub.ac.be/~ktiels/pnlss.html>`_.



In python we need to explicit load the functionality we want to use. Thus we
start by loading the necessary packages.::

  from pyvib.statespace import StateSpace as linss
  from pyvib.statespace import Signal
  from pyvib.pnlss import PNLSS
  from pyvib.common import db
  from pyvib.forcing import multisine
  import numpy as np
  import matplotlib.pyplot as plt

`numpy` provides matrix support(termed arrays no matter the dimension) and
linear algebra. `matplotlib` is a plotting library mimicking the plotting in
matlab.

In pyvib the models are stored as object. This makes it easier to compare
different models in the same script. `StateSpace` is the class for linear state
space models and `PNLSS` is the class for pnlss models.
To create a object we call the class. To call the `StateSpace` class we need to
provide a signal object, which is first created::

  # create signal object
  sig = Signal(u,y)
  sig.set_lines(lines)
  um, ym = sig.average()
  npp, F = sig.npp, sig.F
  R, P = sig.R, sig.P

We provide the measured signals(u,y) which will be used for estimation. Then we
call the method `set_lines` of the object and give the excited lines as input.
The signal object calculate some of the signal properties automatically, as the
'number of points per period' (npp), 'number of excited lines' (F), etc. By
calling `sig.average()`, the signals u,y are averaged over periods P.

A empty linear state space object is created, and when the method
`linmodel.bla(sig)` is called, the 'best linear approximation', total
distortion, and noise distortion are calculated using the signal stored in `sig`.::

  linmodel = linss()
  linmodel.bla(sig)
  models, infodict = linmodel.scan(nvec, maxr)
  lin_errvec = linmodel.extract_model(yval, uval)

`nvec` and `maxr` are specifies the dimension parameters for the subspace
identification. The best model among the identified is extracted by comparing
the error on new, unseen validation data.
Now `linmodel` contains the best subspace model. To see all the variables
contained within `linmodel` run the following in a python interpreter or see
:class:`pyvib.statespace.StateSpace`::

  dir(linmodel)   # show all the variable names
  vars(linmodel)  # show all variables along with their data

The pnlss model is initialized from linmodel. We start by setting the structure
of the model (full, ie. mix of all possible monomials in the state and output
equation for degree 2 and 3).::

  T1 = np.r_[npp, np.r_[0:(R-1)*npp+1:npp]]
  model = PNLSS(linmodel.A, linmodel.B, linmodel.C, linmodel.D)
  model.signal = linmodel.signal
  model.nlterms('x', [2,3], 'full')
  model.nlterms('y', [2,3], 'full')
  model.transient(T1)
  model.optimize(lamb=100, weight=True, nmax=60)

`T1` is the transient settings, ie. how many time samples is prepended each
realization in the averaged time signal `um`. This is done to ensure steady state
of the output, when the model is simulated using `um`. In this case there are two
realizations and each have 1024 points. This gives::

  print(T1)
  >>> array([1024,    0, 1024])

meaning that 1024 samples are prepended and the starting index of the
realizations are 0 and 1014 (remember python is 0-indexed).

To see the difference between the linear and nonlinear model, the output is
calculated using the same data used for estimation. Remember `um` is the
averaged signal calculated earlier::

  tlin, ylin, xlin = linmodel.simulate(um, T1=T1)
  _, ynlin, _ = model.simulate(um)

We now have a pnlss model using at most `nmax=60` Levenberg–Marquardt steps. To
prevent overfitting the model at each step is saved and all models are validated
using unseen validation data; the best performing model is kept. Note that the
transient settings is changed as there is only one realization.::

  nl_errvec = model.extract_model(yval, uval, T1=sig.npp)

`nl_errvec` contains the error of each model.

Finally we plot the linear and nonlinear model error. As found from the
estimation data calculated above::

  plt.figure()
  plt.plot(ym)
  plt.plot(ym-ylin)
  plt.plot(ym-ynlin)
  plt.xlabel('Time index')
  plt.ylabel('Output (errors)')
  plt.legend(('output','linear error','PNLSS error'))
  plt.title('Estimation results')
  figs['estimation_error'] = (plt.gcf(), plt.gca())

`figs` is a dictionary (dict) used for storing the handle and axes for the
figure; the handle is used later for storing the figure to disk. dicts resembles
matlab cells and are a way of storing data using a string as index/key.

