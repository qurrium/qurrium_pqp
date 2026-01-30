"""Qurrium PQP Crossroads - Utilities (:mod:`qurrium_quam_libs.utils`)"""

from qurry import __version__ as qurrium_version
from qurry.process.classical_shadow.utils import (
    convert_to_basis_spin,
    measurements_export,
    measurements_read,
)


def get_qurrium_version_info() -> tuple[int, int, int]:
    """Get the version information of the Qurrium package.

    Returns:
        tuple[int, int, int]: The major, minor, and patch version numbers.
    """
    version_parts = qurrium_version.split(".")[:3]
    version_parts += ["0"] * (3 - len(version_parts))
    return tuple(map(int, version_parts))  # type: ignore


QURRIUM_VERSION = get_qurrium_version_info()
"""The current version of Qurrium."""


__all__ = [
    "convert_to_basis_spin",
    "measurements_export",
    "measurements_read",
    "get_qurrium_version_info",
    "QURRIUM_VERSION",
]
