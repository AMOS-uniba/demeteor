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

    def test_the_bright_stars_are_named(self):
        stars = Catalogue.bundled().stars
        brightest = stars.nsmallest(20, 'vmag')

        assert (brightest.name != 'unnamed').all()
        assert 'Sirius' in set(brightest.name)

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
