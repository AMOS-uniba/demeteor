"""
Fitting a plate without anybody driving it.

The window is for a person deciding which dots are stars. This is for a caller that has already
decided: a plate that was good enough once, a list of dots some camera's own software found, and
nobody to ask. It runs the same Matcher over the same catalogue and reports what it arrived at.

It holds no policy about *quality*. Whether a residual is small enough to keep is the caller's
business -- the caller knows which camera this is and what that camera is normally worth -- so this
reports the residual and says nothing about it. The only failures here are "there is nothing to
fit" and "the optimiser would not run", both of which raise JobError.

The job and the result are plain dictionaries, so a caller can hold them in memory or write them
out as YAML; vasco-fit does the latter and the AMOS server does the former. What comes back is
`amos-reduction/1` documents, which is the format stations already upload, so a fit computed here
and a fit computed at a station arrive at the server by the same road.
"""
import datetime
import logging
import math

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time

from demeteor.catalogue import Catalogue
from demeteor.projections import BorovickaProjection
from demeteor.sensor import DotCollection, SensorData, mask_sparse
from demeteor.matching import Matcher
from demetria.correctors import bandwidth as bandwidth_selection

JOB_FORMAT = 'amos-fit-job/1'
RESULT_FORMAT = 'amos-fit-result/1'
REDUCTION_FORMAT = 'amos-reduction/1'

#: What version stamps the result. The kernel-smoothed correction is a vector field that no twelve
#: numbers can hold, so a caller cannot recompute it and records this instead: the same version,
#: given the same dots and the same baseline, lands on the same field.
try:
    from importlib.metadata import version

    SOFTWARE = f"demeteor {version('demeteor')}"
except Exception:                                       # not installed, running from a checkout
    SOFTWARE = "demeteor (unreleased)"

#: The four constants that are angles, in the order BorovickaProjection takes its twelve. `a` is
#: not one -- it is the fraction by which the radius is stretched -- and neither are the radial
#: constants, which carry radians per millimetre or its powers.
ANGLES = ('a0', 'F', 'epsilon', 'E')

#: demeteor's constructor names, in order. The AMOS documents use lowercase for all twelve.
DEMETEOR = ('x0', 'y0', 'a0', 'A', 'F', 'V', 'S', 'D', 'P', 'Q', 'epsilon', 'E')
DOCUMENT = ('x0', 'y0', 'a0', 'a', 'f', 'v', 's', 'd', 'p', 'q', 'epsilon', 'e')

#: 'auto' means leave-one-out cross-validation over the range below, which is what a bandwidth
#: should be chosen by; a number pins it. It used to be 0.1 and nothing else, inherited from the
#: window's spinbox default -- a number nobody had a reason for, on a night nobody had measured.
DEFAULTS = dict(method='raw', iterations=10000, pre_iterations=0,
                bandwidth='auto', bandwidth_min=0.005, bandwidth_max=2.0, bandwidth_steps=25,
                mask_low=10.0, mask_distant=0.5, sigma_clip=3.0, clip_rounds=2,
                min_stars=8)

log = logging.getLogger('demeteor')


class JobError(ValueError):
    """ The job cannot be read, or names nothing to fit. """


def plate_from_degrees(constants: dict) -> BorovickaProjection:
    """
    A document's twelve into a projection. Degrees in, radians out, for the four that are angles.

    The single crossing between the two conventions on this side, mirroring
    Projection.from_degrees() on the server's. Doing one without the other is a silent error.
    """
    try:
        values = {name: float(constants[key]) for name, key in zip(DEMETEOR, DOCUMENT, strict=True)}
    except (KeyError, TypeError, ValueError) as exc:
        raise JobError(f"the baseline plate cannot be read: {exc}") from exc

    for name in ANGLES:
        values[name] = math.radians(values[name])
    return BorovickaProjection(**values)


def plate_to_degrees(projection: BorovickaProjection) -> dict:
    """
    And back, for a document a person is going to read.

    Plain floats, not numpy scalars: as_tuple() hands back whatever the optimiser put in, and
    yaml.safe_dump refuses an np.float64 outright rather than writing the number.
    """
    values = dict(zip(DEMETEOR, projection.normalised().as_tuple(), strict=True))
    return {key: float(math.degrees(values[name]) if name in ANGLES else values[name])
            for name, key in zip(DEMETEOR, DOCUMENT, strict=True)}


def dots(entries: list, *, fnos: bool) -> DotCollection:
    """
    A job's dot list as a DotCollection. Millimetres, because the caller scaled them.

    Millimetres rather than pixels on purpose: whoever owns the camera's configuration is the
    authority on where the sensor's centre is and how big a pixel is, and the pixel size a station
    writes into its own file is a second opinion that has been seen to disagree with it. Handing the
    collection straight to SensorData and never calling set_shifter_scales leaves it alone.
    """
    if not entries:
        return DotCollection()

    xy = np.array([[float(dot['x']), float(dot['y'])] for dot in entries], dtype=float)
    intensity = np.array([float(dot.get('intensity') or 0) for dot in entries], dtype=float)
    numbers = (np.array([int(dot['fno']) for dot in entries], dtype=int) if fnos else None)
    return DotCollection(xy, intensity, fnos=numbers)


def frame_times(entries: list) -> dict:
    """
    When the caller says each frame was, by frame number, to be handed back untouched.

    The reduction format carries a time per position, and the server checks it against the
    identification's own -- a station whose clock is off has reduced a different moment of sky than
    it identified, which no amount of correct astrometry fixes. Here the two are the same clock, so
    echoing them keeps that check meaningful instead of making it warn on every automatic fit.
    """
    return {int(entry['fno']): entry['time']
            for entry in entries if 'fno' in entry and entry.get('time') is not None}


def build_matcher(job: dict) -> Matcher:
    location = job.get('location') or {}
    try:
        earth = EarthLocation(float(location['longitude']) * u.deg,
                              float(location['latitude']) * u.deg,
                              float(location['altitude']) * u.m)
    except (KeyError, TypeError, ValueError) as exc:
        raise JobError(f"the job names no usable location: {exc}") from exc

    try:
        when = datetime.datetime.fromisoformat(str(job['timestamp']))
    except (KeyError, ValueError) as exc:
        raise JobError(f"the job names no usable timestamp: {exc}") from exc
    if when.tzinfo is None:
        when = when.replace(tzinfo=datetime.UTC)

    stars = dots(job.get('stars') or [], fnos=False)
    meteor = dots(job.get('meteor') or [], fnos=True)
    log.info(f"{stars.count} reference dots, {meteor.count} meteor frames, at {when}")

    sensor = SensorData(stars, meteor, location=earth, timestamp=when,
                        station=str(job.get('station') or 'unknown'))

    # The catalogue demeteor ships, never a file. There is nobody here to choose one, and a
    # setting nobody sets is a setting that goes stale.
    return Matcher(earth, Time(when), catalogue=Catalogue.bundled(), sensor_data=sensor)


def mask(matcher: Matcher, options: dict) -> None:
    """
    Throw away what a person would have thrown away by hand.

    Two limits, both in degrees. `mask_low` drops dots the plate puts near or below the horizon,
    where refraction is large and the roof is in the way. `mask_distant` drops catalogue stars that
    no dot came near, so that the pairing cannot reach for one halfway across the sky.
    """
    if (low := options['mask_low']) is not None:
        altitudes = matcher.sensor_data.stars.project(matcher.projection, masked=True,
                                                      flip_theta=True)[..., 0]
        matcher.mask_sensor_data(
            mask_sparse(matcher.sensor_data.stars, altitudes > math.radians(low)))
        log.info(f"above {low}: {matcher.sensor_data.stars.count_visible} of "
                 f"{matcher.sensor_data.stars.count} dots")

    if (distant := options['mask_distant']) is not None:
        nearest = np.min(matcher.distance_sky_all(masked=True), axis=0)
        matcher.mask_catalogue(
            mask_sparse(matcher.catalogue, nearest < math.radians(distant)))
        log.info(f"within {distant} of a dot: {matcher.catalogue.count_visible} of "
                 f"{matcher.catalogue.count} catalogue objects")


def residuals(matcher: Matcher) -> np.ndarray:
    """ Per-dot angular residual in degrees, for the dots still in the fit. """
    return np.degrees(matcher.position_errors_sky())


def optimise(matcher: Matcher, start: BorovickaProjection, options: dict) -> BorovickaProjection:
    """
    Fit, then throw out the dots that do not belong and fit again.

    The clipping has no counterpart in the window, where a person looks at the plot and masks the
    aeroplane. Without it one dot that is not a star pairs with whatever star is nearest and pulls
    the whole plate towards it -- and there is no one here to notice.

    **This does not converge from a badly wrong baseline, and that is worth knowing.** The pairing
    is only ever recomputed as a side effect of a clipping round, and a fit whose residual is large
    but uniform clips nothing -- so it breaks out after one round and never re-pairs, settling into
    a local minimum with some dots on the wrong stars. Measured on synthetic dots built from a known
    plate: started 1.5 degrees out it lands at 0.83 degrees and stays there whether given 4000
    iterations or 20000. A pre-fit before pairing halves it. Re-pairing between rounds regardless of
    clipping would be the fix, and it is deliberately not done here, because this function was moved
    out of vasco unchanged and changing what it computes belongs in its own commit.
    tests/fitting/test_driver.py pins the present behaviour so that a change to it is visible.
    """
    projection = start

    if options['pre_iterations']:
        projection = BorovickaProjection(
            *matcher.minimize(x0=np.array(projection.as_tuple()),
                              maxiter=options['pre_iterations']))
        matcher.update_pairing()

    for round_number in range(options['clip_rounds'] + 1):
        matcher.update_projection(projection)
        projection = BorovickaProjection(
            *matcher.minimize(x0=np.array(projection.as_tuple()), maxiter=options['iterations'],
                              callback=None))
        matcher.update_projection(projection)

        errors = residuals(matcher)
        log.info(f"round {round_number}: {errors.size} dots, "
                 f"rms {np.degrees(matcher.rms_error(np.radians(errors))):.5f} deg")

        if round_number == options['clip_rounds'] or errors.size == 0:
            break

        limit = options['sigma_clip'] * np.degrees(matcher.rms_error(np.radians(errors)))
        keep = errors <= limit
        if keep.all() or np.count_nonzero(keep) < options['min_stars']:
            log.info("nothing worth clipping" if keep.all()
                     else "clipping would leave too few dots, stopping")
            break

        log.info(f"clipping {np.count_nonzero(~keep)} dots beyond {limit:.5f} deg")
        matcher.mask_sensor_data(mask_sparse(matcher.sensor_data.stars, keep))
        matcher.update_pairing()

    return projection


def bandwidths(matcher: Matcher, options: dict) -> dict:
    """
    A bandwidth for each smoother, chosen or pinned.

    Two of them, and not one shared: the position residuals and the magnitude residuals are
    different fields over the same points, and nothing says the scale over which one varies is the
    scale over which the other does.
    """
    requested = options['bandwidth']
    if requested != 'auto':
        value = float(requested)
        return {'position': value, 'magnitude': value,
                'position_score': None, 'magnitude_score': None, 'chosen': False}

    grid = dict(minimum=options['bandwidth_min'], maximum=options['bandwidth_max'],
                steps=options['bandwidth_steps'])
    position, position_score, _ = bandwidth_selection.select(*matcher.position_smoother_data(),
                                                            **grid)
    magnitude, magnitude_score, _ = bandwidth_selection.select(*matcher.magnitude_smoother_data(),
                                                               **grid)
    return {'position': position, 'magnitude': magnitude,
            'position_score': position_score, 'magnitude_score': magnitude_score, 'chosen': True}


def reduction(matcher: Matcher, projection: BorovickaProjection, job: dict,
              *, method: str, meteor: list, smoothing: dict | None = None) -> dict:
    """
    One amos-reduction/1 document, which is what the server already knows how to read.

    The kernel-smoothed document reports the *parametric* residual, the same number as the raw one,
    and that is deliberate rather than an oversight. The smoother is built from the very residuals
    the fit was left with, so it reproduces them almost exactly and a "residual after smoothing"
    would be near zero by construction -- a measure of how well an interpolator interpolates its
    own training data, not of how good the plate is. The number that means something about this
    camera on this night is the parametric one, and it is also the only one the server can check,
    since the twelve constants are all it stores.
    """
    errors = residuals(matcher)
    magnitudes = matcher.catalogue_vmag(masked=True)

    return {
        'format': REDUCTION_FORMAT,
        'identification': job.get('identification'),
        'software': SOFTWARE,
        'method': method,
        'baseline': job.get('baseline_code'),
        'projection': plate_to_degrees(projection),
        # What it took to get here, for a person and for the next version of this program. The
        # bandwidth in particular: the correction field cannot be stored, so the number it was
        # built with is part of what makes a smoothed reduction reproducible at all.
        'fit': {
            'stars': int(errors.size),
            **({'bandwidth': smoothing['position'],
                'bandwidth_magnitude': smoothing['magnitude'],
                'bandwidth_chosen': smoothing['chosen'],
                **({'bandwidth_loo_mse': smoothing['position_score']}
                   if smoothing['position_score'] is not None else {})}
               if smoothing else {}),
            # Degrees, which is the unit Reduction.quality takes and what the window displays.
            # Matcher works in radians throughout; this is the one place it is converted.
            'residual_rms': float(np.degrees(matcher.rms_error(np.radians(errors))))
            if errors.size else None,
            'residual_max': float(np.max(errors)) if errors.size else None,
            'limiting_magnitude': float(np.max(magnitudes)) if magnitudes.size else None,
        },
        'meteor': meteor,
    }


def meteor_positions(matcher: Matcher, projection: BorovickaProjection, times: dict,
                     *, corrected: bool) -> list:
    """
    Where the meteor was, frame by frame, keyed by the frame number the station gave it.

    By fno and not by position in the list, because DotCollection silently drops a dot whose
    intensity is not positive -- and Kvant reports a negative intensity often enough -- so the
    list is not necessarily the identification's frames in order. The server matches on fno.
    """
    if matcher.sensor_data.meteor.count == 0:
        return []

    matcher.update_projection(projection)
    positions = (matcher.correct_meteor_position(projection) if corrected
                 else matcher.project_meteor(projection))
    magnitudes = (matcher.correct_meteor_magnitude(projection, matcher._calibration) if corrected
                  else matcher._calibration(matcher.sensor_data.meteor.intensities(masked=False)))
    fnos = matcher.sensor_data.meteor.fnos(masked=False)

    return [
        {'fno': int(fno),
         **({'time': times[int(fno)]} if int(fno) in times else {}),
         'alt': float(alt), 'az': float(az % 360.0),
         'magnitude': float(magnitude) if np.isfinite(magnitude) else None}
        for fno, alt, az, magnitude
        in zip(fnos, positions.alt.degree, positions.az.degree, magnitudes, strict=True)
    ]


def fit(job: dict) -> dict:
    if str(job.get('format')) != JOB_FORMAT:
        raise JobError(f"expected a {JOB_FORMAT} job, got {job.get('format')!r}")

    options = {**DEFAULTS, **(job.get('options') or {})}
    baseline = plate_from_degrees(job.get('baseline') or {})

    times = frame_times(job.get('meteor') or [])
    matcher = build_matcher(job)
    if matcher.sensor_data.stars.count < options['min_stars']:
        raise JobError(f"{matcher.sensor_data.stars.count} reference dots is fewer than the "
                       f"{options['min_stars']} this job asks for; there is nothing to fit")

    matcher.update_projection(baseline)
    mask(matcher, options)
    matcher.update_pairing()

    projection = optimise(matcher, baseline, options)

    # 'vasco' and 'vasco-ks' are the method names in the amos-reduction/1 format, which is a
    # contract with everything that reads one -- the AMOS server maps them onto its own choices by
    # exactly these strings. They name the algorithm and its provenance, not the program running it,
    # and they do not change because the code moved house.
    reductions = [reduction(matcher, projection, job, method='vasco',
                           meteor=meteor_positions(matcher, projection, times, corrected=False))]

    if options['method'] == 'kernel':
        # The parametric fit first and the field on top of it, which is why one process does both:
        # the smoother is built from the residuals the fit was left with.
        matcher.update_projection(projection)
        smoothing = bandwidths(matcher, options)
        matcher.update_position_smoother(bandwidth=smoothing['position'])
        matcher.update_magnitude_smoother(bandwidth=smoothing['magnitude'])
        reductions.append(reduction(matcher, projection, job, method='vasco-ks',
                                    meteor=meteor_positions(matcher, projection, times,
                                                            corrected=True),
                                    smoothing=smoothing))

    return {'format': RESULT_FORMAT, 'reductions': reductions}
