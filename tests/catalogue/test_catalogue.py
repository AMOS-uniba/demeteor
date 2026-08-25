import pytest
import datetime

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy import units as u
from astropy.time import Time

from demeteor.catalogue import Catalogue


#: One moment, named once. These tests used to leave the time out and get "now", which made
#: every assertion about a planet an assertion about today -- test_jupiter_is_really_bright would
#: have started failing on its own as Jupiter moved, with nothing having changed.
@pytest.fixture
def when():
    return Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC))


@pytest.fixture
def hyg30(ago, when):
    hyg = Catalogue.bundled()
    hyg.build_planets(ago, when)
    return hyg

@pytest.fixture
def ago():
    return EarthLocation(17.27 * u.deg, 48.37 * u.deg, 531 * u.m)

@pytest.fixture
def altaz(hyg30, ago, when):
    return hyg30.altaz(ago, when, masked=False)


class TestCatalogue:
    def test_load(self, hyg30):
        assert isinstance(hyg30, Catalogue)

    def test_can_count(self, hyg30):
        assert hyg30.count == 5077

    def test_with_planets(self, hyg30, ago, when):
        assert len(hyg30.altaz(ago, when, masked=False)) == 5077

    def test_polaris(self, hyg30, ago, when):
        altaz = hyg30.altaz(ago, when, masked=False)
        # Check that the altitude of Polaris is within 1 degree of the observer's latitude
        assert altaz[54].alt.degree == pytest.approx(ago.lat.degree, abs=1)
        # Check that the azimuth of Polaris is within 1 degree of north (on northern hemisphere)
        assert abs((altaz[54].az.degree + 180) % 360 - 180) == pytest.approx(0, abs=1)

    def test_sirius(self, hyg30, ago, when):
        altaz = hyg30.altaz(ago, when, masked=False)
        assert altaz[7].alt.degree == pytest.approx(-25.6, abs=0.2)
        assert altaz[7].az.degree == pytest.approx(86.6, abs=0.2)

    def test_planets(self, hyg30, ago, when):
        hyg30.build_planets(ago, when)
        assert len(hyg30.planets_skycoord) == 7

    def test_jupiter_is_really_bright(self, hyg30, ago, when):
        assert hyg30.vmag(ago, when, masked=False)[3] == pytest.approx(-2, abs=0.5)

    def test_vmag(self, hyg30, ago, when):
        vmag = hyg30.vmag(ago, when, masked=False)
        assert vmag.shape == (5077,)
        assert vmag[7] == pytest.approx(-1.45, abs=0.05)

    def test_altaz_is_a_skycoord(self, altaz):
        assert isinstance(altaz, SkyCoord)
