"""
The unit disk, which is where an all-sky frame is compared with the sky.

A projection maps sensor millimetres to a zenith distance and an azimuth. Neither of those is a
convenient thing to draw, to difference or to smooth over, because the azimuth wraps and the two
have different units. So everything in between happens on a disk: the zenith at the origin, the
horizon at radius one, azimuth going round. On that disk a residual is a plain two-component vector
and a field over it is a field over a square-ish domain, which is what a kernel smoother wants.

Azimuth zero points down, hence the minus on every y and the quarter turn added back on the way
out. That is a display convention that leaked into the arithmetic long ago and is now load-bearing:
`disk_to_altaz(altaz_to_disk(x)) == x` only holds because both ends agree about it.
"""
import math
from typing import Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz
from numpy.typing import NDArray

from demeteor.catalogue import Catalogue

#: A quarter turn. The horizon is this far from the zenith, and the disk's radius is one, so this
#: is also the scale factor between an angle and a radius.
QuarterTau = math.tau / 4


def altaz_to_disk(altaz: Optional[AltAz]) -> NDArray:
    """ astropy's AltAz onto the disk. """
    if altaz is None:
        return np.empty(shape=(0, 2))

    return np.stack(
        (
            np.sin(altaz.az.radian) * (QuarterTau - altaz.alt.radian) / QuarterTau,
            -np.cos(altaz.az.radian) * (QuarterTau - altaz.alt.radian) / QuarterTau,
        ), axis=1,
    )


def numpy_to_disk(altaz: NDArray) -> NDArray:
    """ The same, for an (N, 2) array of (altitude, azimuth) in radians. """
    return np.stack(
        [
            np.sin(altaz[..., 1]) * (QuarterTau - altaz[..., 0]) / QuarterTau,
            -np.cos(altaz[..., 1]) * (QuarterTau - altaz[..., 0]) / QuarterTau,
        ], axis=1,
    )


def disk_to_altaz(xy: NDArray) -> AltAz:
    """ And back off the disk, as an AltAz. """
    return AltAz(
        (QuarterTau + np.arctan2(xy[..., 1], xy[..., 0])) * u.rad,
        (1 - np.sqrt(xy[..., 0] ** 2 + xy[..., 1] ** 2)) * QuarterTau * u.rad,
    )


def proj_to_disk(obs: Optional[NDArray]) -> NDArray:
    """
    A projection's own output onto the disk.

    Takes (zenith distance, azimuth) as a projection returns it -- not altitude -- which is why
    this is not numpy_to_disk with the arguments the other way round.
    """
    if obs is None:
        return np.empty(shape=(0, 2))

    z, a = obs.T
    return np.stack((z * np.sin(a) / math.tau * 4,
                     -z * np.cos(a) / math.tau * 4), axis=1)


def mask_sparse(array: Catalogue, sparse_mask: NDArray) -> NDArray:
    """
    Widen a mask over the visible items into a mask over all of them.

    A caller that computes something per *visible* object -- a residual, say -- gets an array as
    long as the visible ones, and cannot use it as a mask over the whole collection. This puts each
    flag back where it came from and leaves everything already hidden hidden.

    Works on anything with `.mask` and `.count`, which is both a Catalogue and a SensorData's dot
    collections; the annotation names only the one this package can import without a cycle.
    """
    mask = array.mask.nonzero()[0]
    idx = np.zeros(array.count, dtype=bool)
    idx[mask[sparse_mask]] = True
    return idx
