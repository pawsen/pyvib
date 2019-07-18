.. currentmodule:: pyvib

Contributing
============

I happily accept contributions.

If you wish to add a new feature or fix a bug:

#. `Check for open issues <https://github.com/pawsen/pyvib/issues>`_ or open
   a fresh issue to start a discussion around a feature idea or a bug.
#. Fork the `pyvib repository on Github <https://github.com/pawsen/pyvib>`_
   to start making your changes.
#. Write a test which shows that the bug was fixed or that the feature works
   as expected.
#. Send a pull request and bug the maintainer until it gets merged and published.
   :) Make sure to add yourself to ``CONTRIBUTORS.txt``.


Setting up your development environment
---------------------------------------

It is recommended, and even enforced by the make file, that you use a
`virtualenv
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_::

  $ python3 -m venv venv3
  $ source venv3/bin/activate
  $ pip install -r dev-requirements.txt


Running the tests
-----------------

Run the test suite to ensure no additional problem arises. Our ``Makefile``
handles much of this for you as long as you're running it `inside of a
virtualenv <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_::

  $ make test
  [... magically installs dependencies and runs tests on your virtualenv]
  Ran 182 tests in 1.633s

  OK (SKIP=6)
