import platform
import datetime

__all__ = ['get_sys_dict', 'system_info']


def get_sys_dict():
    """
    Test which packages are installed on system.
    Returns
    -------
    sys_prop : `dict`
        A dictionary containing the programs and versions installed on this
        machine
    """

    try:
        from pyvib.version import version as pyvib_version
        from pyvib.version import githash as pyvib_git_description
    except ImportError:
        pyvib_version = 'Missing version.py; re-run setup.py'
        pyvib_git_description = 'N/A'

    # Dependencies
    try:
        from numpy import __version__ as numpy_version
    except ImportError:
        numpy_version = "NOT INSTALLED"

    try:
        from scipy import __version__ as scipy_version
    except ImportError:
        scipy_version = "NOT INSTALLED"

    try:
        from matplotlib import __version__ as matplotlib_version
    except ImportError:
        matplotlib_version = "NOT INSTALLED"

    try:
        from pandas import __version__ as pandas_version
    except ImportError:
        pandas_version = "NOT INSTALLED"

    try:
        from bs4 import __version__ as bs4_version
    except ImportError:
        bs4_version = "NOT INSTALLED"

    try:
        from PyQt4.QtCore import PYQT_VERSION_STR as pyqt_version
    except ImportError:
        pyqt_version = "NOT INSTALLED"

    try:
        from zeep import __version__ as zeep_version
    except ImportError:
        zeep_version = "NOT INSTALLED"

    try:
        from sqlalchemy import __version__ as sqlalchemy_version
    except ImportError:
        sqlalchemy_version = "NOT INSTALLED"

    sys_prop = {'Time': datetime.datetime.utcnow().strftime("%A, %d. %B %Y %I:%M%p UT"),
                'System': platform.system(), 'Processor': platform.processor(),
                'Pyvib': pyvib_version, 'Pyvib_git': pyvib_git_description,
                'Arch': platform.architecture()[0], "Python": platform.python_version(),
                'NumPy': numpy_version,
                'SciPy': scipy_version, 'matplotlib': matplotlib_version,
                'Pandas': pandas_version,
                'beautifulsoup': bs4_version, 'PyQt': pyqt_version,
                'Zeep': zeep_version, 'Sqlalchemy': sqlalchemy_version
                }
    return sys_prop


def system_info():
    """
    Takes dictionary from sys_info() and prints the contents in an attractive fashion.
    """
    sys_prop = get_sys_dict()

    # title
    print("==========================================================")
    print(" Pyvib Installation Information\n")
    print("==========================================================\n")

    # general properties
    print("###########")
    print(" General")
    print("###########")
    # OS and architecture information

    for sys_info in ['Time', 'System', 'Processor', 'Arch', 'Pyvib', 'Pyvib_git']:
        print('{0} : {1}'.format(sys_info, sys_prop[sys_info]))

    if sys_prop['System'] == "Linux":
        distro = " ".join(platform.linux_distribution())
        print("OS: {0} (Linux {1} {2})".format(distro, platform.release(), sys_prop['Processor']))
    elif sys_prop['System'] == "Darwin":
        print("OS: Mac OS X {0} ({1})".format(platform.mac_ver()[0], sys_prop['Processor']))
    elif sys_prop['System'] == "Windows":
        print("OS: Windows {0} {1} ({2})".format(platform.release(), platform.version(),
                                                 sys_prop['Processor']))
    else:
        print("Unknown OS ({0})".format(sys_prop['Processor']))

    print("\n")
    # required libraries
    print("###########")
    print(" Required Libraries ")
    print("###########")

    for sys_info in ['Python', 'NumPy', 'SciPy', 'matplotlib']:
        print('{0}: {1}'.format(sys_info, sys_prop[sys_info]))

    print("\n")

    # recommended
    print("###########")
    print(" Recommended Libraries ")
    print("###########")

    for sys_info in ['beautifulsoup', 'PyQt', 'Zeep', 'Sqlalchemy', 'Pandas']:
        print('{0}: {1}'.format(sys_info, sys_prop[sys_info]))
