"""
Which dot is which star.

The pairing is the part of a match that can be wrong without anything looking wrong: a plate fitted
against the wrong stars still has a residual, still converges, and still produces a number somebody
will read. The driver tests exercise it end to end; these ask it directly, on dots placed where a
known plate says a known star should be, so that the right answer is known in advance.
"""
import datetime

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time

from demeteor.catalogue import Catalogue
from demeteor.matching import Matcher
from demeteor.projections import BorovickaProjection
from demeteor.sensor import DotCollection, SensorData

WHERE = EarthLocation(17.273933 * u.deg, 48.372763 * u.deg, 580.0 * u.m)
WHEN = Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC))

#: A plate to place dots with. Any would do; this one is roughly a real all-sky.
PLATE = BorovickaProjection(x0=0.05, y0=-0.03, a0=np.radians(100.0), A=1e-4, F=np.radians(65.0),
                            V=0.42, S=-0.08, D=0.13, P=0.0, Q=0.0,
                            epsilon=np.radians(1.5), E=np.radians(210.0))


@pytest.fixture(scope='module')
def catalogue():
    return Catalogue.bundled()


@pytest.fixture(scope='module')
def bright(catalogue):
    """ The indices of the brightest stars well above the horizon, and where they are. """
    altaz = catalogue.altaz(WHERE, WHEN, masked=False)
    vmag = catalogue.vmag(WHERE, WHEN, masked=False)
    chosen = np.where((altaz.alt.degree > 40) & (vmag < 2.5))[0]
    assert chosen.size >= 5, f"only {chosen.size} bright stars up; the fixture needs a few"
    return chosen, altaz


def matcher_for(indices, altaz, *, plate=PLATE, jitter=0.0, seed=1):
    """ A Matcher whose dots are exactly where `plate` says those stars are. """
    alt = altaz.alt.radian[indices]
    az = altaz.az.radian[indices]
    x, y = plate.invert(np.pi / 2 - alt, az)

    if jitter:
        generator = np.random.default_rng(seed)
        x = x + generator.normal(0, jitter, x.shape)
        y = y + generator.normal(0, jitter, y.shape)

    dots = DotCollection(np.stack([x, y], axis=1), np.full(x.shape, 1000.0))
    sensor = SensorData(dots, DotCollection(), location=WHERE, timestamp=WHEN.to_datetime())
    matcher = Matcher(WHERE, WHEN, catalogue=Catalogue.bundled(), sensor_data=sensor)
    matcher.update_projection(plate)
    return matcher


class TestItPairsWithTheRightStar:
    def test_a_dot_on_a_star_is_paired_to_that_star(self, bright):
        indices, altaz = bright

        matcher = matcher_for(indices, altaz)

        assert list(matcher.pairing) == list(indices)

    def test_and_survives_a_nudge_the_size_of_a_pixel(self, bright):
        """
        Worth knowing the scale before choosing a number: V is 0.42 radians per millimetre, so a
        millimetre of sensor is twenty-four degrees of sky and one 4.4 micrometre pixel is about a
        tenth of a degree. That is a realistic centroiding error and leaves the pairing alone. A
        tenth of a *millimetre* would be 2.4 degrees, further than the typical star is from its
        neighbour, and would pair with a different star -- correctly.
        """
        indices, altaz = bright

        matcher = matcher_for(indices, altaz, jitter=0.0044)

        assert list(matcher.pairing) == list(indices)

    def test_the_residual_of_a_perfect_placement_is_zero(self, bright):
        indices, altaz = bright

        matcher = matcher_for(indices, altaz)

        assert matcher.rms_error(matcher.position_errors_sky()) == pytest.approx(0, abs=1e-9)

    def test_a_wrong_plate_pairs_wrongly_and_says_nothing_about_it(self, bright):
        """
        The failure this file exists for. Place the dots with one plate, match with another a few
        degrees away, and some dots pair with a neighbour instead -- and the only sign is a residual
        that is larger than it should be, which is indistinguishable from a bad night.
        """
        indices, altaz = bright
        askew = BorovickaProjection(x0=0.05, y0=-0.03, a0=np.radians(115.0), A=1e-4,
                                    F=np.radians(65.0), V=0.42, S=-0.08, D=0.13, P=0.0, Q=0.0,
                                    epsilon=np.radians(1.5), E=np.radians(210.0))

        matcher = matcher_for(indices, altaz, plate=PLATE)
        matcher.update_projection(askew)

        assert list(matcher.pairing) != list(indices)
        assert np.isfinite(matcher.rms_error(matcher.position_errors_sky()))


class TestChangingThingsRepairs:
    def test_a_new_projection_repairs_by_default(self, bright):
        indices, altaz = bright
        matcher = matcher_for(indices, altaz)
        before = list(matcher.pairing)

        matcher.update_projection(BorovickaProjection(V=0.42, a0=np.radians(280.0)))

        assert list(matcher.pairing) != before

    def test_unless_the_pairing_is_fixed(self, bright):
        """ Which is how the window lets a person pin a pairing they have corrected by hand. """
        indices, altaz = bright
        matcher = matcher_for(indices, altaz)
        before = list(matcher.pairing)

        matcher.fix_pairing(True)
        matcher.update_projection(BorovickaProjection(V=0.42, a0=np.radians(280.0)))

        assert list(matcher.pairing) == before


class TestMasking:
    def test_masking_the_catalogue_narrows_what_can_be_paired(self, bright):
        indices, altaz = bright
        matcher = matcher_for(indices, altaz)

        keep = np.zeros(matcher.catalogue.count, dtype=bool)
        keep[indices] = True
        matcher.mask_catalogue(keep)

        assert matcher.catalogue.count_visible == len(indices)
        assert list(matcher.pairing) == list(indices), "still the right ones, from fewer candidates"

    def test_masking_the_dots_narrows_the_residuals(self, bright):
        indices, altaz = bright
        matcher = matcher_for(indices, altaz)

        keep = np.ones(matcher.sensor_data.stars.count, dtype=bool)
        keep[0] = False
        matcher.mask_sensor_data(keep)

        assert matcher.position_errors_sky().size == len(indices) - 1


class TestTheFit:
    def test_minimize_leaves_the_plate_alone_when_it_is_already_right(self, bright):
        """
        Started at the answer, a fit should stay near it. Not exactly at it -- Nelder-Mead moves
        before it decides not to -- but the residual must not get worse.
        """
        indices, altaz = bright
        matcher = matcher_for(indices, altaz)
        before = matcher.rms_error(matcher.position_errors_sky())

        fitted = BorovickaProjection(*matcher.minimize(x0=np.array(PLATE.as_tuple()), maxiter=200))
        matcher.update_projection(fitted)

        assert matcher.rms_error(matcher.position_errors_sky()) == pytest.approx(before, abs=1e-6)

    def test_a_projection_can_be_read_back(self, bright):
        """ There was a setter and no getter, so every caller reached for _projection. """
        indices, altaz = bright
        matcher = matcher_for(indices, altaz)

        assert matcher.projection.as_tuple() == PLATE.as_tuple()
