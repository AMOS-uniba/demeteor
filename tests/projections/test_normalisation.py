"""
Bringing a fit's angles back into the range they mean.

An optimiser is unconstrained in these parameters and nothing wraps them afterwards, so the same
plate can come out as a0 = 12.7 rad or with a negative epsilon. Every caller was left to cope with
that alone, which is how vasco came to write a fit into spinboxes that clamped it -- 107 degrees of
sky, silently.

What matters here is that normalising is never a change. The tests project through both forms and
compare, rather than reasoning about the algebra, because the one case that is easy to get wrong
looks entirely reasonable: flipping the sign of epsilon without moving E by half a turn keeps every
parameter in range and moves the sky by about twice the zenith distance.
"""
import math

import numpy as np
import pytest

from demeteor.projections import BorovickaProjection
from demeteor.projections.base import TAU, normalise_angles
from demeteor.projections.koniferka import KoniferkaProjection

#: A real fitted plate, from AGO's calibration file
KY = (0.014864, -0.014848, 4.4094506287729995, -0.002302, 1.4911492386711016, 0.452072,
      -0.000868, 1.6e-05, 1.074161, -0.002488, 0.013736282265263492, 2.613772659569206)

#: The whole sensor, in millimetres
GRID = np.meshgrid(np.linspace(-6, 6, 40), np.linspace(-6, 6, 40))
XS, YS = (axis.ravel() for axis in GRID)


def sky_distance(one, other) -> float:
    """ The furthest apart two projections put any point of the sensor, in radians. """
    z1, a1 = one(XS, YS)
    z2, a2 = other(XS, YS)
    azimuth = (a1 - a2 + math.pi) % TAU - math.pi
    return float(np.nanmax(np.hypot(z1 - z2, np.sin(z1) * azimuth)))


def replaced(index: int, value: float) -> tuple:
    return KY[:index] + (value,) + KY[index + 1:]


A0, F, EPSILON, E = 2, 4, 10, 11


class TestNormaliseAngles:
    """ The arithmetic on its own. """
    def test_a_plate_in_range_is_untouched(self):
        assert normalise_angles(1.2, 3.4, 0.5, 6.0) == (1.2, 3.4, 0.5, 6.0)

    @pytest.mark.parametrize('turns', [-3, -1, 1, 2, 7])
    def test_whole_turns_come_off(self, turns):
        a0, f, epsilon, e = normalise_angles(1.2 + turns * TAU, 3.4 + turns * TAU,
                                             0.5, 6.0 + turns * TAU)
        assert a0 == pytest.approx(1.2)
        assert f == pytest.approx(3.4)
        assert epsilon == pytest.approx(0.5)
        assert e == pytest.approx(6.0)

    def test_a_negative_zenith_distance_moves_its_azimuth_by_half_a_turn(self):
        _, _, epsilon, e = normalise_angles(0.0, 0.0, -0.5, 1.0)

        assert epsilon == pytest.approx(0.5)
        assert e == pytest.approx(1.0 + math.pi)

    def test_a_reflex_zenith_distance_is_folded_the_same_way(self):
        _, _, epsilon, e = normalise_angles(0.0, 0.0, TAU - 0.5, 1.0)

        assert epsilon == pytest.approx(0.5)
        assert e == pytest.approx(1.0 + math.pi)

    @pytest.mark.parametrize('epsilon', [-9.0, -0.5, 0.0, 0.5, math.pi, 4.0, 9.0, 100.0])
    def test_the_zenith_distance_ends_within_half_a_turn(self, epsilon):
        _, _, epsilon, _ = normalise_angles(0.0, 0.0, epsilon, 0.0)

        assert 0 <= epsilon <= math.pi

    @pytest.mark.parametrize('angle', [-100.0, -TAU, -0.001, 0.0, 3.0, TAU, 100.0])
    def test_the_azimuths_end_within_a_whole_turn(self, angle):
        a0, f, _, e = normalise_angles(angle, angle, 0.5, angle)

        for value in (a0, f, e):
            assert 0 <= value < TAU


class TestNormalisedIsNotAChange:
    """ And that applying it leaves the projection pointing at exactly the same sky. """
    #: Anything larger is the arithmetic, not the wrapping: 1e-12 rad is 2e-7 arcsec.
    TOLERANCE = 1e-12

    @pytest.mark.parametrize('written', [
        pytest.param(KY, id='already in range'),
        pytest.param(replaced(A0, KY[A0] + 2 * TAU), id='a0 past two turns'),
        pytest.param(replaced(A0, KY[A0] - 5 * TAU), id='a0 below zero'),
        pytest.param(replaced(F, KY[F] - 3 * TAU), id='F below zero'),
        pytest.param(replaced(E, KY[E] - TAU), id='E below zero'),
        pytest.param(replaced(EPSILON, -KY[EPSILON]), id='epsilon negative'),
        pytest.param(replaced(EPSILON, TAU - KY[EPSILON]), id='epsilon reflex'),
        pytest.param(replaced(EPSILON, 0.0), id='no zenith shift at all'),
        pytest.param((0.014864, -0.014848, -12.0, -0.002302, 20.0, 0.452072,
                      -0.000868, 1.6e-05, 1.074161, -0.002488, -8.0, -19.0),
                     id='everything at once'),
    ])
    def test_the_sky_does_not_move(self, written):
        projection = BorovickaProjection(*written)

        assert sky_distance(projection, projection.normalised()) < self.TOLERANCE

    @pytest.mark.parametrize('written', [
        pytest.param(replaced(A0, KY[A0] + 2 * TAU), id='a0'),
        pytest.param(replaced(EPSILON, -KY[EPSILON]), id='epsilon'),
    ])
    def test_the_result_is_in_range(self, written):
        _, _, a0, _, f, _, _, _, _, _, epsilon, e = BorovickaProjection(*written).normalised().as_tuple()

        assert 0 <= a0 < TAU
        assert 0 <= f < TAU
        assert 0 <= e < TAU
        assert 0 <= epsilon <= math.pi

    @pytest.mark.parametrize('epsilon', [0.0137, 0.1234, 0.5])
    def test_flipping_the_zenith_distance_without_the_half_turn_would_move_the_sky(self, epsilon):
        """
        The reason normalise_angles takes all four together and not one at a time.

        The error is roughly twice the zenith distance -- 0.055 rad on this plate, whose epsilon is
        0.0137, and 2.0 rad at half a radian -- so it is small only where the shift is small, and
        it is never the tolerance above.
        """
        fitted = BorovickaProjection(*replaced(EPSILON, -epsilon))
        naive = BorovickaProjection(*replaced(EPSILON, epsilon))

        assert sky_distance(fitted, naive) > 2 * epsilon

    def test_it_does_not_mutate_the_original(self):
        projection = BorovickaProjection(*replaced(A0, KY[A0] + TAU))

        projection.normalised()

        assert projection.as_tuple()[A0] == pytest.approx(KY[A0] + TAU)

    def test_normalising_twice_changes_nothing_further(self):
        once = BorovickaProjection(*replaced(A0, KY[A0] - 4 * TAU)).normalised()

        assert once.normalised().as_tuple() == once.as_tuple()

    def test_the_constants_that_are_not_angles_are_untouched(self):
        original = BorovickaProjection(*replaced(A0, KY[A0] + TAU))
        normalised = original.normalised()

        for index in (0, 1, 3, 5, 6, 7, 8, 9):
            assert normalised.as_tuple()[index] == original.as_tuple()[index]

    def test_a_projection_with_no_angles_normalises_to_itself(self):
        from demeteor.projections import EquidistantProjection

        projection = EquidistantProjection()

        assert projection.normalised() is projection


class TestKoniferkaNormalisation:
    #: Small radial coefficients, so that the transform stays inside a hemisphere and the
    #: comparison measures the wrapping rather than the overflow of an exponential
    SANE = (0.01, -0.01, 8.0, -0.0023, -9.0, 0.45, 0.001, 20.0, 0.001, 20.0, -0.3, 7.0)

    def test_the_sky_does_not_move(self):
        projection = KoniferkaProjection(*self.SANE)

        assert sky_distance(projection, projection.normalised()) < 1e-12

    def test_it_folds_the_zenith_distance_the_same_way(self):
        normalised = KoniferkaProjection(*self.SANE).normalised().as_tuple()

        assert normalised[EPSILON] == pytest.approx(0.3)
        assert normalised[E] == pytest.approx((7.0 + math.pi) % TAU)
