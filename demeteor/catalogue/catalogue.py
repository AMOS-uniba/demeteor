import datetime
from importlib import resources
from typing import Any, Optional

import numpy as np
import pandas as pd

from pathlib import Path

from astropy.coordinates import EarthLocation, FK5, SkyCoord, AltAz, get_body, concatenate, Angle
from astropy.time import Time
from astropy import units as u
from numpy.ma.core import shape


class Catalogue:
    COLUMNS = ['ra', 'dec', 'dist', 'vmag', 'absmag', 'name']
    PLANETS = ['mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

    #: The catalogue that ships with demeteor: HYG 4.4 down to visual magnitude 6, which is roughly
    #: what an all-sky camera sees, with proper names for the 428 stars that have one. It is by
    #: astronexus and licensed CC BY-SA 4.0 -- a copyleft, so anything adapted from it stays under
    #: that licence. See data/README.md beside the file for the attribution and the changes made,
    #: and tools/build_catalogue.py for how it is regenerated from a new HYG release.
    #:
    #: It lives here rather than in each program that needs it because the column list above is an
    #: assertion, and a copy of the file kept somewhere else is a copy that can fall out of step
    #: with it -- which is exactly what happened: adding `name` here broke every caller still
    #: shipping the five-column HYG 3.0 export, and each of them had its own.
    BUNDLED = 'HYG44.tsv'

    @classmethod
    def bundled(cls) -> 'Catalogue':
        """
        The catalogue shipped with demeteor, whatever the working directory is.

        Read as a stream rather than by path so that it also works from a zipped install, where
        there is no file on disk to point at.
        """
        with resources.files(__package__).joinpath('data', cls.BUNDLED).open('rb') as handle:
            return cls(handle)

    def __init__(self, filename: Path = None):
        # Empty until someone says where and when. Constructing a catalogue used to run a
        # seven-body ephemeris at latitude zero, longitude zero, at whatever moment the process
        # happened to start -- work nobody asked for, at coordinates nobody meant.
        self.planets = pd.DataFrame(columns=self.COLUMNS)
        self.planets_skycoord = None

        if filename is None:
            self._populated = False
            self.stars = pd.DataFrame(columns=self.COLUMNS)
            self.stars_skycoord = SkyCoord([] * u.deg,
                                           [] * u.deg,
                                           frame=FK5(equinox=Time('J2000')))
        else:
            self._populated = True
            self.stars = pd.read_csv(filename, sep='\t', header=1)
            self.stars_skycoord = SkyCoord(self.stars.ra.to_numpy() * u.deg,
                                           self.stars.dec.to_numpy() * u.deg,
                                           frame=FK5(equinox=Time('J2000')))

        assert self.stars.columns.tolist() == self.COLUMNS, \
            f"Invalid format for columns, expected {self.COLUMNS}"

        # len(PLANETS) and not len(self.planets): the planets are seven whether or not their
        # positions have been worked out yet, and the mask has to be the right length from the
        # start or the first caller to set one gets an assertion about a shape.
        self._mask = np.ones(shape=len(self.stars) + len(self.PLANETS), dtype=bool)

    @property
    def populated(self) -> bool:
        return self._populated

    def build_planets(self,
                      location: EarthLocation,
                      time: Time):
        """
        Work out where the seven planets are, and how bright, as seen from here and now.

        Both arguments are required, and that is the point. They used to default to latitude zero
        and to the moment of the call, which is never what anybody means and is impossible to
        notice: a planet given the wrong time still comes out somewhere plausible with a plausible
        magnitude. It cost four call sites across three programs, each of which asked altaz() for a
        past moment and then vmag() for none, so that a planet was drawn where it stood then and
        sized by how bright it is today -- Venus by 1.6 magnitudes, a factor of four in flux.
        """
        sun = get_body('sun', time=time, location=location)

        planets = []
        for name in self.PLANETS:
            body = get_body(name, time=time, location=location)
            sundist = body.hcrs.distance
            phase = body.separation(sun)
            planets.append(pd.DataFrame(
                data=[
                    [
                        body.ra.degree,
                        body.dec.degree,
                        body.distance.to(u.lightyear).value,
                        self.planet_brightness(name, body.distance, sundist, phase),
                        -10,
                        name.title(),
                    ]
                ],
                columns=self.COLUMNS,
            ))

        self.planets = pd.concat(planets)
        self.planets_skycoord = SkyCoord(self.planets.ra.to_numpy() * u.deg,
                                         self.planets.dec.to_numpy() * u.deg,
                                         frame=FK5(equinox=Time('J2000')))

    def radec(self,
              location: EarthLocation,
              time: Time,
              *,
              planets: bool = True,
              masked: bool) -> SkyCoord:
        """
        Equatorial coordinates of everything, planets first and then stars.

        `time` is required even though a star's right ascension does not depend on it: the planets'
        do, and a signature that lets one caller omit it is a signature that lets the wrong caller
        omit it.
        """
        if planets:
            self.build_planets(location, time)
            return (np.concatenate([self.planets_skycoord, self.stars_skycoord])[self.mask]
                    if masked else
                    np.concatenate([self.planets_skycoord, self.stars_skycoord]))

        # Stars alone, so the planets' seven entries come off the front of the mask too. Without
        # that this raised: the mask is always as long as stars plus planets, and indexing 5077
        # stars with 5084 booleans is not a subtle failure but it is a rare one, since nothing
        # in this ecosystem asks for the stars without the planets.
        return self.stars_skycoord[self.mask[len(self.PLANETS):]] if masked else self.stars_skycoord

    def altaz(self,
              location: EarthLocation,
              time: Time,
              *,
              planets: bool = True,
              masked: bool) -> AltAz:
        """
        Return the catalogue in alt-az coordinates at `location` and at `time`.
        Optionally include planets.
        """
        altaz = AltAz(location=location, obstime=time, pressure=100000 * u.pascal, obswl=550 * u.nm)
        radec = self.radec(location, time, planets=planets, masked=masked)
        return radec.transform_to(altaz)

    def vmag(self,
             location: EarthLocation,
             time: Time,
             *,
             masked: bool) -> np.ndarray[float]:
        """
        Visual magnitudes of everything, planets first and then stars, as seen from here and then.

        Always includes the planets, whatever the name of the parameter this docstring used to
        claim. Their brightness is what `time` is for -- a star's is a column in the file.
        """
        self.build_planets(location, time)
        vmags = pd.concat([self.planets, self.stars]).vmag.to_numpy(dtype=float)
        return vmags[self.mask] if masked else vmags

    def names(self, *, masked: bool) -> np.ndarray:
        """
        Return the names of all objects, in the same order as vmag() and radec().

        Planets first and then stars, which is the order everything else in this class uses and
        the reason this lives here rather than in the callers: an index into `mask` or `count`
        means nothing without it, and a caller that assumed stars came first would put the wrong
        name on every object it labelled.

        A star with no proper name in the catalogue reads as an em dash; the planets are
        capitalised.

        Needs no location and no time, and now genuinely does not: it used to read the names out of
        the planet table, which meant it silently depended on an ephemeris having been run, and
        gave nothing at all before one had been. A planet's name is a planet's name.
        """
        names = np.concatenate([
            np.array([planet.title() for planet in self.PLANETS], dtype=object),
            self.stars.name.to_numpy(),
        ])
        return names[self.mask] if masked else names

    @staticmethod
    def planet_brightness(planet: str,
                          distance_earth: u.Quantity,
                          distance_sun: u.Quantity,
                          phase: Angle):
        """
        Get the approximate visual magnitude of a planet.
        Shamelessly stolen from APC, Montenbruck 1999
    """
        p = phase.degree / 100.0

        match planet:
            case 'mercury':
                mag = -0.42 + (3.80 - (2.73 - 2 * p) * p) * p
            case 'venus':
                mag = -4.40 + (0.09 + (2.39 - 0.65 * p) * p) * p
            case 'mars':
                mag = -1.52 + 1.6 * p
            case 'jupiter':
                mag = -9.4 + 0.5 * p
            case 'saturn':
                # Currently we do not care about the rings, but it might be worth checking it later
                sd = 0 # np.abs(np.sin(lat))
                dl = 0 # np.abs((dlong + np.pi) % (2 * np.pi) - np.pi) / 100
                mag = -8.88 + 2.60 * sd + 1.25 * sd**2 + 4.4 * dl
            case 'uranus':
                mag = -7.19
            case 'neptune':
                mag = -6.87

        return mag + 5 * np.log10(distance_earth.to(u.au).value * distance_sun.to(u.au).value)

    @property
    def count(self) -> int:
        """ Stars plus the seven planets, whether or not an ephemeris has been run for them. """
        return len(self.stars) + len(self.PLANETS)

    @property
    def count_visible(self) -> int:
        return len(self.mask[self.mask])

    @property
    def mask(self) -> np.ndarray[bool]:
        return self._mask

    @mask.setter
    def mask(self, m: Optional[np.ndarray[bool]] = None) -> None:
        self._mask = np.ones(shape=(self.count,), dtype=bool) if m is None else m
        assert self.mask.shape == (self.count,), \
            f"Mask shape does not match data shape: expected {self.count,}, got {self.mask.shape}"

    def __str__(self):
        return f"<Catalogue with {self.count_visible} / {self.count} reference objects>"
