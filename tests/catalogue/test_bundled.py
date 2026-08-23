"""
The catalogue demeteor ships.

COLUMNS is an assertion, so a copy of the data file kept anywhere else is a copy that can fall out
of step with it. That is not hypothetical: adding `name` to the list broke every program still
shipping the five-column HYG 3.0 export, and the server, vasco and vamos each had their own. The
file lives in the package now, and these tests are what keep the two together.
"""
import io

import pytest

from demeteor.catalogue import Catalogue


class TestBundled:
    def test_it_loads_without_being_told_where_it_is(self):
        catalogue = Catalogue.bundled()

        assert len(catalogue.stars) > 5000

    def test_it_loads_from_anywhere(self, tmp_path, monkeypatch):
        """ The old fixture was a relative path and only worked from the repository root. """
        monkeypatch.chdir(tmp_path)

        assert len(Catalogue.bundled().stars) > 5000

    def test_it_matches_the_columns_this_class_asserts_on(self):
        assert Catalogue.bundled().stars.columns.tolist() == Catalogue.COLUMNS

    def test_it_reaches_naked_eye_magnitude(self):
        stars = Catalogue.bundled().stars

        assert stars.vmag.max() == pytest.approx(6.0, abs=0.05)
        assert stars.vmag.min() < -1, "Sirius should be in there"

    @pytest.mark.parametrize('star', ['Sirius', 'Canopus', 'Arcturus', 'Vega', 'Capella',
                                      'Rigel', 'Betelgeuse', 'Procyon', 'Altair', 'Polaris'])
    def test_the_stars_a_person_can_point_at_are_named(self, star):
        """
        Names are here to identify what is visible by eye, so the test is about those rather than
        about a count. Not every bright row has one: the 0.96 entry 15 arcsec from Capella is its
        companion, and HYG 4.4 gives the name to the primary and leaves the secondary blank.
        """
        assert star in set(Catalogue.bundled().stars.name)

    def test_most_of_the_brightest_are_named(self):
        brightest = Catalogue.bundled().stars.nsmallest(30, 'vmag')

        assert (brightest.name != 'unnamed').sum() >= 25

    def test_it_is_the_same_catalogue_every_time(self):
        assert len(Catalogue.bundled().stars) == len(Catalogue.bundled().stars)

    def test_an_empty_catalogue_still_has_no_stars(self):
        """
        Catalogue() means no stars, and callers rely on it as a placeholder -- vasco holds one
        until a catalogue is chosen. Its `count` is 7 rather than 0, because the planets are built
        either way; that is what count means.
        """
        empty = Catalogue()

        assert len(empty.stars) == 0
        assert empty.count == len(Catalogue.PLANETS)


class TestTheOldFormatIsRefused:
    """
    The five-column HYG 3.0 export, which several programs were still carrying.

    Refusing it loudly is the right behaviour -- a catalogue silently missing a column would put
    the wrong names on stars -- but the message is what someone will have to act on, so it says
    what was expected.
    """
    OLD = "# ra dec\nra\tdec\tdist\tvmag\tabsmag\n101.287215\t-16.716116\t2.6371\t-1.44\t1.454\n"

    def test_it_raises(self):
        with pytest.raises(AssertionError):
            Catalogue(io.StringIO(self.OLD))

    def test_it_says_what_it_wanted(self):
        with pytest.raises(AssertionError, match="name"):
            Catalogue(io.StringIO(self.OLD))


class TestAttribution:
    """
    The catalogue is CC BY-SA 4.0, so the notice has to travel with the data.

    Worth a test rather than trusting the build: if packaging ever stops including non-Python files
    the data would go too and this would fail loudly, but a change that keeps the .tsv and drops
    the .md would leave us redistributing someone's work without the attribution it requires.
    """
    @staticmethod
    def notice() -> str:
        from importlib import resources

        return (resources.files('demeteor.catalogue')
                .joinpath('data', 'README.md').read_text(encoding='utf-8'))

    def test_it_ships_beside_the_data(self):
        assert len(self.notice()) > 500

    @pytest.mark.parametrize('required', [
        'HYG',                                              # what it is
        'astronexus',                                       # who made it
        'CC BY-SA 4.0',                                     # under what terms
        'creativecommons.org/licenses/by-sa/4.0',           # where to read them
        'codeberg.org/astronexus/hyg',                       # where it came from
    ])
    def test_it_carries_what_the_licence_asks_for(self, required):
        assert required in self.notice()

    def test_it_says_the_file_was_modified(self):
        """ BY-SA 4.0 3(a)(1)(B): indicate if you modified the material. This file is a subset. """
        notice = self.notice()

        assert 'modified' in notice.lower()
        assert 'Changes made to it' in notice


class TestEncoding:
    """
    The catalogue is UTF-8 and some names are not ASCII.

    Worth pinning: the file this replaced carried `Yunü` as `YunÃ¼`, which is UTF-8 read as
    Latin-1. It survives a build only if every step -- reading HYG, writing the tsv, reading it
    back -- agrees on the encoding, and none of them says so out loud.
    """
    def test_a_non_ascii_name_survives(self):
        names = set(Catalogue.bundled().stars.name)

        assert 'Yunü' in names

    def test_and_is_not_mojibaked(self):
        names = set(Catalogue.bundled().stars.name)

        assert 'YunÃ¼' not in names

    def test_every_name_decodes(self):
        for name in Catalogue.bundled().stars.name:
            assert isinstance(name, str), name


class TestNames:
    """
    Naming an object by its index into the catalogue.

    The index space is planets first and then stars, which is not obvious and is exactly the sort
    of thing a caller gets wrong: vasco pairs a sensor star with an index into `mask`, and a lookup
    that assumed stars came first would label every one of them wrongly.
    """
    def test_it_is_as_long_as_the_catalogue(self):
        catalogue = Catalogue.bundled()

        assert len(catalogue.names(masked=False)) == catalogue.count

    def test_the_planets_come_first(self):
        names = Catalogue.bundled().names(masked=False)

        assert list(names[:len(Catalogue.PLANETS)]) == [p.title() for p in Catalogue.PLANETS]

    def test_the_stars_follow_in_magnitude_order(self):
        catalogue = Catalogue.bundled()
        names = catalogue.names(masked=False)

        assert names[len(Catalogue.PLANETS)] == 'Sirius'

    def test_it_lines_up_with_the_magnitudes(self):
        """ The two are read together, so an off-by-seven in either would show up here. """
        from astropy.coordinates import EarthLocation
        import astropy.units as u

        catalogue = Catalogue.bundled()
        where = EarthLocation(17.27 * u.deg, 48.37 * u.deg, 531 * u.m)
        names = catalogue.names(masked=False)
        vmags = catalogue.vmag(where, masked=False)

        assert len(names) == len(vmags)
        brightest_star = vmags[len(Catalogue.PLANETS):].argmin() + len(Catalogue.PLANETS)
        assert names[brightest_star] == 'Sirius'

    def test_masking_shortens_it(self):
        catalogue = Catalogue.bundled()

        assert len(catalogue.names(masked=True)) <= len(catalogue.names(masked=False))
