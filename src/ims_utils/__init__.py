"""Spectrum utilities."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ims-utils")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"
