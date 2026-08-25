"""
A catalogue never guesses when you are looking.

`build_planets`, `radec`, `altaz` and `vmag` all used to default `time` to the moment of the call.
That is never what anybody means and it cannot be noticed: a planet given the wrong time still
comes out somewhere plausible with a plausible magnitude, and the stars -- which are almost all of
the catalogue and all of what anyone checks -- do not move at all.

It cost four call sites in three programs, each of which asked altaz() for a past moment and then
vmag() for none, so that a planet was placed where it stood then and given the brightness it has
today. Over eleven months that is Venus by 1.6 magnitudes, a factor of four in flux.

So the argument is required, and these are the tests that keep it required.
"""
import datetime
import inspect

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time

from demeteor.catalogue import Catalogue

WHERE = EarthLocation(17.27 * u.deg, 48.37 * u.deg, 531 * u.m)
WHEN = Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC))
LATER = Time(datetime.datetime(2025, 8, 25, 21, 56, 37, tzinfo=datetime.UTC))


@pytest.fixture
def catalogue():
    return Catalogue.bundled()


class TestTheTimeIsRequired:
    @pytest.mark.parametrize('method', ['build_planets', 'radec', 'altaz', 'vmag'])
    def test_it_has_no_default(self, method):
        """
        Asserted on the signature and not by calling, so that a default reintroduced for one method
        and not the others is still caught -- and so that the reason is in the failure message.
        """
        parameter = inspect.signature(getattr(Catalogue, method)).parameters['time']

        assert parameter.default is inspect.Parameter.empty, \
            f"Catalogue.{method}() must not guess the time"

    @pytest.mark.parametrize('method', ['radec', 'altaz', 'vmag'])
    def test_omitting_it_raises(self, catalogue, method):
        with pytest.raises(TypeError):
            getattr(catalogue, method)(WHERE, masked=False)

    def test_the_location_is_required_too(self):
        parameter = inspect.signature(Catalogue.build_planets).parameters['location']

        assert parameter.default is inspect.Parameter.empty


class TestItActuallyDependsOnTheTime:
    """ If it did not, requiring the argument would be ceremony. """
    def test_the_planets_move(self, catalogue):
        first = catalogue.radec(WHERE, WHEN, masked=False)[:len(Catalogue.PLANETS)]
        second = catalogue.radec(WHERE, LATER, masked=False)[:len(Catalogue.PLANETS)]

        assert np.all(first.separation(second).degree > 1)

    def test_and_change_brightness(self, catalogue):
        first = catalogue.vmag(WHERE, WHEN, masked=False)[:len(Catalogue.PLANETS)]
        second = catalogue.vmag(WHERE, LATER, masked=False)[:len(Catalogue.PLANETS)]

        # Venus is the one that swings, being close and showing phases
        assert np.max(np.abs(first - second)) > 0.5

    def test_the_stars_do_not(self, catalogue):
        """ Which is exactly why nothing ever looked wrong. """
        planets = len(Catalogue.PLANETS)
        first = catalogue.vmag(WHERE, WHEN, masked=False)[planets:]
        second = catalogue.vmag(WHERE, LATER, masked=False)[planets:]

        assert np.array_equal(first, second)


class TestConstructionRunsNoEphemeris:
    def test_a_new_catalogue_has_no_planet_positions(self, catalogue):
        """
        It used to compute seven bodies at latitude zero, longitude zero, at whatever moment the
        process happened to start -- work nobody asked for at coordinates nobody meant.
        """
        assert catalogue.planets_skycoord is None
        assert len(catalogue.planets) == 0

    def test_but_it_still_counts_them(self, catalogue):
        """ There are seven planets whether or not anyone has worked out where they are, and the
            mask has to be the right length before the first caller sets one. """
        assert catalogue.count == len(catalogue.stars) + len(Catalogue.PLANETS)
        assert catalogue.mask.shape == (catalogue.count,)

    def test_and_still_names_them(self, catalogue):
        """ A planet's name does not depend on where or when you look, and now nor does reading
            it: this used to come back empty until an ephemeris had been run. """
        names = list(catalogue.names(masked=False)[:len(Catalogue.PLANETS)])

        assert names == [planet.title() for planet in Catalogue.PLANETS]

    def test_a_mask_set_before_any_ephemeris_still_fits_afterwards(self, catalogue):
        catalogue.mask = np.ones(catalogue.count, dtype=bool)
        catalogue.build_planets(WHERE, WHEN)

        assert len(catalogue.vmag(WHERE, WHEN, masked=True)) == catalogue.count


class TestStarsWithoutPlanets:
    def test_it_no_longer_indexes_past_the_end(self, catalogue):
        """
        planets=False returns the stars alone, so the planets' seven entries have to come off the
        front of the mask as well. They did not, and indexing 5070 stars with 5077 booleans is a
        hard failure -- a rare one only because nothing in this ecosystem asks for it.
        """
        stars = catalogue.radec(WHERE, WHEN, planets=False, masked=True)

        assert len(stars) == len(catalogue.stars)

    def test_and_a_mask_is_honoured(self, catalogue):
        mask = np.ones(catalogue.count, dtype=bool)
        mask[-4:] = False
        catalogue.mask = mask

        assert len(catalogue.radec(WHERE, WHEN, planets=False, masked=True)) \
            == len(catalogue.stars) - 4
