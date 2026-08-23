"""
Writing a projection down and reading it back.

Three separate slips lived here, all of the same shape: one wrong letter in a positional list of
twelve. from_dotmap read P from S, Koniferka's as_tuple reported p1 twice and never p2, and
as_dict emitted x/y where every reader wants x0/y0 -- so a projection demeteor wrote could not be
read back by demeteor, which is what vasco-cli writes its calibration files with.

The way to catch that class of thing is to round trip with twelve values that are all different,
so that a slot taking its neighbour's value cannot look like a pass.
"""
import io
import math
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from demeteor.projections import BorovickaProjection, EquidistantProjection, Projection
from demeteor.projections.koniferka import KoniferkaProjection

#: Twelve distinct values, in the order both classes take their parameters
DISTINCT = (0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.11, 1.22, 1.33)

BOROVICKA_NAMES = ('x0', 'y0', 'a0', 'A', 'F', 'V', 'S', 'D', 'P', 'Q', 'epsilon', 'E')
KONIFERKA_NAMES = ('x0', 'y0', 'a0', 'A', 'F', 'V', 'p1', 'r1', 'p2', 'r2', 'epsilon', 'E')


class TestAsTuple:
    @pytest.mark.parametrize('cls', [BorovickaProjection, KoniferkaProjection])
    def test_every_parameter_comes_back_in_its_own_place(self, cls):
        assert cls(*DISTINCT).as_tuple() == DISTINCT

    def test_it_survives_being_fed_back_in(self, ):
        once = BorovickaProjection(*DISTINCT).as_tuple()

        assert BorovickaProjection(*once).as_tuple() == once


class TestAsDict:
    @pytest.mark.parametrize('cls, names', [(BorovickaProjection, BOROVICKA_NAMES),
                                            (KoniferkaProjection, KONIFERKA_NAMES)])
    def test_it_names_the_parameters_the_way_the_constructor_does(self, cls, names):
        assert set(cls(*DISTINCT).as_dict()) == set(names)

    @pytest.mark.parametrize('cls', [BorovickaProjection, KoniferkaProjection])
    def test_what_it_writes_can_be_read_back(self, cls):
        """ The x/y against x0/y0 mismatch: what demeteor wrote, demeteor could not load. """
        written = cls(*DISTINCT).as_dict()

        assert cls(**written).as_tuple() == DISTINCT

    def test_the_shared_azimuth_is_written_once_and_consistently(self):
        # E belongs to both the tilt shifter and the zenith shifter, so the merged dict must not
        # end up with two different ideas of it
        projection = BorovickaProjection(*DISTINCT)

        assert projection.as_dict()['E'] == DISTINCT[-1]

    def test_a_projection_with_no_parameters_still_round_trips(self):
        assert EquidistantProjection(**EquidistantProjection().as_dict()) is not None


class TestFromDotmap:
    """ How a calibration file becomes a projection. `dm` is only ever attribute-accessed. """
    def test_every_parameter_is_read_from_its_own_name(self):
        loaded = BorovickaProjection.from_dotmap(
            SimpleNamespace(**dict(zip(BOROVICKA_NAMES, DISTINCT))))

        assert loaded.as_tuple() == DISTINCT

    def test_koniferka_reads_its_own_names(self):
        loaded = KoniferkaProjection.from_dotmap(
            SimpleNamespace(**dict(zip(KONIFERKA_NAMES, DISTINCT))))

        assert loaded.as_tuple() == DISTINCT

    def test_a_projection_survives_the_whole_way_out_and_back(self):
        """ as_dict is what vasco-cli writes into a calibration file; from_dotmap is what reads it. """
        original = BorovickaProjection(*DISTINCT)

        reloaded = BorovickaProjection.from_dotmap(SimpleNamespace(**original.as_dict()))

        assert reloaded.as_tuple() == original.as_tuple()


class TestFromDict:
    def test_a_registered_projection_is_rebuilt_by_name(self):
        original = BorovickaProjection(*DISTINCT)

        rebuilt = Projection.from_dict(dict(name=BorovickaProjection.name,
                                            parameters=original.as_dict()))

        assert isinstance(rebuilt, BorovickaProjection)
        assert rebuilt.as_tuple() == DISTINCT


class TestSilence:
    """ A library that prints is a library you cannot call in a loop. """
    @pytest.mark.parametrize('cls, params', [
        (BorovickaProjection, (0.01, -0.01, 1.2, -0.002, 2.4, 0.45, 0.001, 0.02, 0.001, 0.02,
                               0.3, 1.9)),
        (KoniferkaProjection, (0.01, -0.01, 1.2, -0.002, 2.4, 0.45, 0.001, 20.0, 0.001, 20.0,
                               0.3, 1.9)),
    ])
    def test_projecting_and_inverting_say_nothing(self, cls, params):
        projection = cls(*params)
        captured = io.StringIO()

        with redirect_stdout(captured):
            projection.invert(*projection(1.0, 2.0))

        assert captured.getvalue() == ''

    @pytest.mark.parametrize('cls, params', [
        (BorovickaProjection, (0.01, -0.01, 1.2, -0.002, 2.4, 0.45, 0.001, 0.02, 0.001, 0.02,
                               0.3, 1.9)),
        (KoniferkaProjection, (0.01, -0.01, 1.2, -0.002, 2.4, 0.45, 0.001, 20.0, 0.001, 20.0,
                               0.3, 1.9)),
    ])
    def test_and_still_invert(self, cls, params):
        projection = cls(*params)

        x, y = projection.invert(*projection(1.0, 2.0))

        assert (x, y) == pytest.approx((1.0, 2.0), abs=1e-9)


class TestBounds:
    """ What the fitter is allowed to try. """
    @pytest.mark.parametrize('cls', [BorovickaProjection, KoniferkaProjection])
    def test_the_zenith_distance_is_unbounded(self, cls):
        """
        It used to be bounded below at zero, which is a wall exactly where an all-sky camera sits.
        A negative value is the same plate as its positive twin with E turned around, so the fit
        may pass through and be normalised afterwards.
        """
        low, high = cls.bounds[10]

        assert low is None and high is None

    @pytest.mark.parametrize('cls', [BorovickaProjection, KoniferkaProjection])
    def test_the_radial_scale_is_still_bounded_away_from_zero(self, cls):
        # This one is not a wrapping question: a projection with V <= 0 does not exist
        low, _ = cls.bounds[5]

        assert low is not None and low > 0

    @pytest.mark.parametrize('cls', [BorovickaProjection, KoniferkaProjection])
    def test_there_is_one_bound_per_parameter(self, cls):
        assert len(cls.bounds) == len(DISTINCT)


class TestTiltAmplitude:
    """ A is a ratio, not an angle -- the bound is where the inverse stops working. """
    def test_the_radius_is_stretched_by_exactly_that_fraction(self):
        from demeteor.projections.shifters import TiltShifter
        import numpy as np

        amplitude, rho = 0.01, 4.0
        tilt = TiltShifter(x0=0, y0=0, a0=0, A=amplitude, F=0, E=0)
        theta = np.linspace(0, math.tau, 17)[:-1]

        r, _ = tilt(rho * np.cos(theta), rho * np.sin(theta))

        assert r / rho == pytest.approx(1 + amplitude * np.sin(theta), abs=1e-15)

    @pytest.mark.parametrize('amplitude', [-1.5, 1.5, 100.0])
    def test_an_amplitude_that_cannot_be_inverted_is_refused(self, amplitude):
        from demeteor.projections.shifters import TiltShifter

        with pytest.raises(AssertionError):
            TiltShifter(x0=0, y0=0, a0=0, A=amplitude, F=0, E=0)

    @pytest.mark.parametrize('amplitude', [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_anything_within_the_bound_is_accepted(self, amplitude):
        from demeteor.projections.shifters import TiltShifter

        assert TiltShifter(x0=0, y0=0, a0=0, A=amplitude, F=0, E=0).A == amplitude
