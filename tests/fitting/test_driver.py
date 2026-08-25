"""
Does the fit find the plate it was given dots from?

Neither vasco nor the AMOS server ever asked this. Both checked a fit against a *real* night, where
the truth is unknown and the residual is the only evidence — good enough to notice a regression,
useless for telling a fit that works from one that happens to land somewhere plausible.

Here the truth is known because the dots were made from it: take a plate, take real stars from the
catalogue, invert the plate to find where each would fall on the sensor, and hand those back as a
job. From the plate itself the fit is then exact, which is the floor everything else is measured
against -- and from a plate 1.5 degrees off it is not, which is the finding this file exists for.
"""
import datetime
import math

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time

from demeteor.catalogue import Catalogue
from demeteor.fitting import JobError, fit
from demeteor.fitting.driver import DOCUMENT, plate_from_degrees, plate_to_degrees
from demeteor.projections import BorovickaProjection

WHERE = dict(latitude=48.372763, longitude=17.273933, altitude=580.0)
WHEN = datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC)

#: A plausible all-sky plate, in the degrees a document carries
TRUTH = dict(x0=0.05, y0=-0.03, a0=100.0, a=1e-4, f=65.0,
             v=0.42, s=-0.08, d=0.13, p=0.0, q=0.0, epsilon=1.5, e=210.0)


def sensor_dots(plate: BorovickaProjection, location, time, *, above=25.0):
    """
    Where the catalogue's brightest stars would land on the sensor under this plate.

    Inverted rather than projected, so the dots are exactly consistent with the plate by
    construction and any residual the fit is left with is the fit's own.
    """
    catalogue = Catalogue.bundled()
    altaz = catalogue.altaz(location, time, masked=False)
    vmag = catalogue.vmag(location, time, masked=False)

    visible = (altaz.alt.degree > above) & (vmag < 4.0)
    alt = altaz.alt.radian[visible]
    az = altaz.az.radian[visible]

    x, y = plate.invert(np.pi / 2 - alt, az)
    inside = np.isfinite(x) & np.isfinite(y) & (np.hypot(x, y) < 6.0)

    return [{'x': float(a), 'y': float(b), 'intensity': 1000.0}
            for a, b in zip(x[inside], y[inside], strict=True)]


def job_for(stars, baseline, **options):
    return {
        'format': 'amos-fit-job/1',
        'identification': 'synthetic.yaml',
        'station': 'TEST',
        'timestamp': WHEN.strftime('%Y-%m-%d %H:%M:%S.%f'),
        'location': dict(WHERE),
        'baseline': baseline,
        'baseline_code': 'truth',
        'stars': stars,
        'meteor': [],
        'options': {'iterations': 4000, 'clip_rounds': 0, **options},
    }


@pytest.fixture(scope='module')
def truth():
    return plate_from_degrees(TRUTH)


@pytest.fixture(scope='module')
def stars(truth):
    location = EarthLocation(WHERE['longitude'] * u.deg, WHERE['latitude'] * u.deg,
                             WHERE['altitude'] * u.m)
    dots = sensor_dots(truth, location, Time(WHEN))
    assert len(dots) > 30, f"only {len(dots)} dots to fit; the test needs a populated sky"
    return dots


class TestItRecoversThePlate:
    def test_from_the_truth_the_fit_is_exact(self, stars):
        """
        The floor, and it is a real zero: dots built by inverting this plate are refitted by it to
        the last digit. Anything above this on real data is the camera, the centroiding or the
        catalogue -- not the arithmetic here.
        """
        report = fit(job_for(stars, dict(TRUTH)))['reductions'][0]['fit']

        assert report['residual_rms'] < 1e-6
        assert report['stars'] == len(stars)

    def test_a_drifted_baseline_is_improved_but_not_recovered(self, stars):
        """
        Where the method's reach ends, pinned so that moving it is visible.

        Started 1.5 degrees out in a0 and 2% out in V, the fit lands at about 0.8 degrees and stays
        there: 4000 iterations and 20000 give the same number to five places, so it is not an
        iteration budget but a local minimum with some dots paired to the wrong stars. optimise()
        re-pairs only as a side effect of a clipping round, and with a residual this uniform nothing
        exceeds sigma_clip times the RMS, so it breaks out after one round and never re-pairs. A
        pre-fit before pairing (`pre_iterations`) roughly halves it, to 0.35.

        1.5 degrees is a deliberately unreasonable start. Twelve parameters over a whole sky do not
        get searched blind, and no real caller starts blind: a camera that has been pointed at the
        sky has been fitted once. What this pins is the size of the drift the refinement tolerates,
        not a defect waiting on a fix.
        """
        wrong = dict(TRUTH, a0=TRUTH['a0'] + 1.5, v=TRUTH['v'] * 1.02)

        plain = fit(job_for(stars, wrong, mask_distant=None))['reductions'][0]['fit']
        prefit = fit(job_for(stars, wrong, mask_distant=None,
                             iterations=20000, pre_iterations=200))['reductions'][0]['fit']

        assert plain['residual_rms'] == pytest.approx(0.83, abs=0.1)
        assert prefit['residual_rms'] < plain['residual_rms']

    def test_masking_distant_stars_assumes_the_baseline_is_already_close(self, stars):
        """
        mask_distant drops catalogue stars that no dot came near, which keeps the pairing from
        reaching across the sky -- and requires the starting plate to be within about that distance.
        At the default half a degree, a baseline 1.5 degrees out fits to 12 degrees; at two degrees,
        to 0.8. Worth knowing when reading a bad result: the limit that protects a good baseline is
        also what a bad one runs into first, and past a point it removes the catalogue entirely,
        which is a JobError rather than a poor fit. TestTheCatalogueCanBeMaskedToNothing covers that
        end.
        """
        wrong = dict(TRUTH, a0=TRUTH['a0'] + 1.5, v=TRUTH['v'] * 1.02)

        tight = fit(job_for(stars, wrong, mask_distant=0.5))['reductions'][0]['fit']
        loose = fit(job_for(stars, wrong, mask_distant=2.0))['reductions'][0]['fit']

        assert tight['residual_rms'] > 5.0
        assert loose['residual_rms'] < 1.0


class TestTheCatalogueCanBeMaskedToNothing:
    """
    The one way a plate too far out crashed rather than complained.

    `mask_distant` keeps the catalogue stars some dot came near. Far enough out, that is none of
    them, and pairing 58 dots against an empty catalogue used to raise
    `IndexError: index -1 is out of bounds for axis 0 with size 0` from two layers down in Matcher
    -- an empty array indexed with a sentinel that nothing consumed. It is a job that cannot be
    done, so it says so.
    """

    #: Far enough that no catalogue star is within any sensible mask_distant of any dot, without
    #: being so far that the sensor leaves the sky altogether.
    HOPELESS = dict(TRUTH, a0=TRUTH['a0'] + 40.0)

    def test_it_is_a_job_error_and_not_an_index_error(self, stars):
        with pytest.raises(JobError):
            fit(job_for(stars, self.HOPELESS, mask_distant=0.05))

    def test_it_says_which_limit_and_how_far_out_it_actually_was(self, stars):
        """ The two numbers that separate "the plate is stale" from "the limit is too tight". """
        with pytest.raises(JobError, match=r'mask_distant=0\.05') as caught:
            fit(job_for(stars, self.HOPELESS, mask_distant=0.05))

        assert 'nearest' in str(caught.value)

    def test_without_the_limit_the_same_baseline_merely_fits_badly(self, stars):
        """
        The guard is the mask's, not the fit's: hopeless is allowed to be hopeless. This is what
        tells the two apart, and it is why the refusal names mask_distant rather than the baseline.
        """
        report = fit(job_for(stars, self.HOPELESS, mask_distant=None))['reductions'][0]['fit']

        assert report['residual_rms'] > 1.0

    def test_the_dots_can_empty_out_the_same_way(self, stars):
        """
        The mirror case, one line earlier in mask(): mask_low can drop every dot, and then the very
        next statement reduced an empty axis and raised ValueError. min_stars does not catch it --
        that counts the dots the job arrived with, before any mask.
        """
        with pytest.raises(JobError, match=r'mask_low=89\.5'):
            fit(job_for(stars, dict(TRUTH), mask_low=89.5))

    def test_a_generous_limit_keeps_stars_and_does_not_raise(self, stars):
        """ So that the guard cannot be satisfied by refusing everything. """
        report = fit(job_for(stars, dict(TRUTH), mask_distant=0.5))['reductions'][0]['fit']

        assert report['residual_rms'] < 1e-6


class TestTheDocument:
    def test_it_is_a_reduction_the_ecosystem_can_read(self, stars):
        result = fit(job_for(stars, dict(TRUTH)))

        assert result['format'] == 'amos-fit-result/1'
        assert len(result['reductions']) == 1, "raw only, unless kernel is asked for"
        document = result['reductions'][0]
        assert document['format'] == 'amos-reduction/1'
        assert document['method'] == 'vasco', "the method name is a wire contract, not a program"
        assert document['identification'] == 'synthetic.yaml'
        assert sorted(document['projection']) == sorted(DOCUMENT)

    def test_the_kernel_pass_adds_a_second_and_chooses_a_bandwidth(self, stars):
        """
        And on this data it chooses the largest bandwidth it is offered, which is the right answer
        and worth asserting for it: the dots were built from the plate exactly, so what is left over
        is optimiser noise with no spatial structure at all. A smoother of no particular width is
        the same as no smoother, and leave-one-out declines to invent a field that is not there.
        On a real night, where the residual does have structure, it lands in the interior -- 0.047
        on the night this was written against.
        """
        result = fit(job_for(stars, dict(TRUTH), method='kernel'))

        assert [d['method'] for d in result['reductions']] == ['vasco', 'vasco-ks']
        smoothed = result['reductions'][1]['fit']
        assert smoothed['bandwidth_chosen'] is True
        assert smoothed['bandwidth'] == pytest.approx(2.0), "the top of the grid: no field to find"
        assert smoothed['bandwidth_magnitude'] > 0

    def test_a_pinned_bandwidth_is_used_and_marked(self, stars):
        result = fit(job_for(stars, dict(TRUTH), method='kernel', bandwidth=0.25))
        smoothed = result['reductions'][1]['fit']

        assert smoothed['bandwidth'] == 0.25
        assert smoothed['bandwidth_chosen'] is False

    def test_every_number_is_a_plain_float(self, stars):
        """
        yaml.safe_dump refuses an np.float64 outright, and a representer error at the very end
        would throw away a fit that had already succeeded.
        """
        result = fit(job_for(stars, dict(TRUTH), method='kernel'))

        for document in result['reductions']:
            for value in document['projection'].values():
                assert type(value) is float
            for key, value in document['fit'].items():
                assert value is None or type(value) in (bool, int, float), f"{key} is {type(value)}"


class TestDegreesAndRadians:
    def test_the_crossing_is_a_round_trip(self):
        there = plate_from_degrees(TRUTH)
        back = plate_to_degrees(there)

        for name in DOCUMENT:
            assert back[name] == pytest.approx(TRUTH[name], abs=1e-9), name

    def test_the_four_angles_are_converted_and_the_rest_are_not(self):
        plate = plate_from_degrees(TRUTH)

        assert plate.axis_shifter.a0 == pytest.approx(math.radians(TRUTH['a0']))
        assert plate.zenith_shifter.epsilon == pytest.approx(math.radians(TRUTH['epsilon']))
        assert plate.radial_transform.V == pytest.approx(TRUTH['v']), "not an angle"
        assert plate.axis_shifter.A == pytest.approx(TRUTH['a']), "a ratio, not an angle"


class TestRefusals:
    def test_a_job_of_the_wrong_format(self, stars):
        with pytest.raises(JobError, match='amos-fit-job/1'):
            fit(dict(job_for(stars, dict(TRUTH)), format='something-else/9'))

    def test_too_few_stars(self, stars):
        with pytest.raises(JobError, match='nothing to fit'):
            fit(job_for(stars[:3], dict(TRUTH)))

    def test_an_unreadable_baseline(self, stars):
        with pytest.raises(JobError, match='baseline'):
            fit(job_for(stars, {'x0': 0.0}))

    def test_no_location(self, stars):
        job = job_for(stars, dict(TRUTH))
        del job['location']['latitude']

        with pytest.raises(JobError, match='location'):
            fit(job)

    def test_no_timestamp(self, stars):
        job = job_for(stars, dict(TRUTH))
        del job['timestamp']

        with pytest.raises(JobError, match='timestamp'):
            fit(job)
