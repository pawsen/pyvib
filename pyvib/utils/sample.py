# -*- coding: utf-8 -*-
"""pyvib sample data files"""
import socket
import os.path
import warnings
from shutil import move
from tempfile import TemporaryDirectory
from subprocess import check_call
from .config import get_and_create_sample_dir


__all__ = ['download_sample_data', 'get_sample_file']

# https://api.github.com/repos/pawsen/pyvib_data/contents/pyvib/data/nlbeam
# https://stackoverflow.com/a/18194523/1121523
#_github_downloader = 'https://minhaskamal.github.io/DownGit/#/home?url='
_base_urls = (
    'https://api.github.com/repos/pawsen/pyvib_data/contents/pyvib/data/',
    #'https://github.com/pawsen/pyvib_data/tree/master/pyvib/data/',
)


# files or folders to download
sample_files = {
    "NLBEAM": "nlbeam",
    "2DOF": "2dof",
    "BOUCWEN": "boucwen",
    "SILVERBOX": "silverbox",
}

def download_sample_data(show_progress=True):
    """
    Download all sample data at once. This will overwrite any existing files.
    Parameters
    ----------
    show_progress: `bool`
        Show a progress bar during download
    Returns
    -------
    None
    """
    for filename in sample_files.values():
        get_sample_file(filename, url_list=_base_urls, overwrite=True)


def get_sample_file(filename, url_list=_base_urls, overwrite=False):
    """
    Downloads a sample file. Will download  a sample data file and move it to
    the sample data directory. Also, uncompresses zip files if necessary.
    Returns the local file if exists.
    Parameters
    ----------
    filename: `str`
        Name of the file
    url_list: `str` or `list`
        urls where to look for the file
    show_progress: `bool`
        Show a progress bar during download
    overwrite: `bool`
        If True download and overwrite an existing file.
    timeout: `float`
        The timeout in seconds. If `None` the default timeout is used from
        `astropy.utils.data.Conf.remote_timeout`.
    Returns
    -------
    result: `str`
        The local path of the file. None if it failed.
    """

    # Creating the directory for sample files to be downloaded
    sampledata_dir = get_and_create_sample_dir()
    src = os.path.join(sampledata_dir, filename)
    if not overwrite and os.path.isfile(src):
        return src
    else:
        # check each provided url to find the file
        for base_url in url_list:
            try:
                url = base_url + filename
                with TemporaryDirectory() as d:
                    rc = check_call(['github-download.sh', url], cwd=d)
                    # move files to the data directory
                    move(d, src)
                return src
            except (socket.error, socket.timeout) as e:
                warnings.warn("Download failed with error {}. \n"
                              "Retrying with different mirror.".format(e))
        # if reach here then file has not been downloaded.
        warnings.warn("File {} not found.".format(filename))
    return None
