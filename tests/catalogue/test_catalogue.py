import pytest
import datetime

from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy import units as u
from astropy.time import Time

from amosutils.catalogue import Catalogue


@pytest.fixture
def hyg30(ago):
    hyg = Catalogue('tests/HYG30.tsv')
    hyg.build_planets(ago)
    return hyg

@pytest.fixture
def ago():
    return EarthLocation(17.27 * u.deg, 48.37 * u.deg, 531 * u.m)

@pytest.fixture
def altaz(hyg30, ago):
    return hyg30.altaz(ago, Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC)), masked=False)


class TestCatalogue:
    def test_load(self, hyg30):
        assert isinstance(hyg30, Catalogue)

    def test_can_count(self, hyg30):
        assert hyg30.count == 5075

    def test_with_planets(self, hyg30, ago):
        assert len(hyg30.altaz(ago, masked=False)) == 5075

    def test_polaris(self, hyg30, ago):
        altaz = hyg30.altaz(ago, masked=False)
        # Check that the altitude of Polaris is within 1 degree of the observer's latitude
        assert altaz[47].alt.degree == pytest.approx(ago.lat.degree, abs=1)
        # Check that the azimuth of Polaris is within 1 degree of north (on northern hemisphere)
        assert abs((altaz[47].az.degree + 180) % 360 - 180) == pytest.approx(0, abs=1)

    def test_sirius(self, hyg30, ago):
        altaz = hyg30.altaz(ago, Time(datetime.datetime(2024, 9, 25, 21, 56, 37, tzinfo=datetime.UTC)), masked=False)
        assert altaz[0].alt.degree == pytest.approx(-25.6, abs=0.2)
        assert altaz[0].az.degree == pytest.approx(86.6, abs=0.2)

    def test_planets(self, hyg30, ago):
        hyg30.build_planets(ago)
        assert len(hyg30.planets_skycoord) == 7

    def test_jupiter_is_really_bright(self, hyg30, ago):
        assert hyg30.vmag(ago, masked=False)[5071] == pytest.approx(-2, abs=0.5)

    def test_vmag(self, hyg30, ago):
        vmag = hyg30.vmag(ago, masked=False)
        assert vmag.shape == (5075,)
        assert vmag[0] == pytest.approx(-1.45, abs=0.05)

    def test_altaz_is_a_skycoord(self, altaz):
        assert isinstance(altaz, SkyCoord)
