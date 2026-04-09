import numpy as np

import pytest

from demeteor.projections.transformers import LinearTransformer, ExponentialTransformer, BiexponentialTransformer, \
    SaneBiexponentialTransformer, SaneExponentialTransformer


def pytest_generate_tests(metafunc):
    if hasattr(metafunc.cls, 'params'):
        funcarglist = metafunc.cls.params.get(metafunc.function.__name__, None)

        if funcarglist:
            argnames = sorted(funcarglist[0])
            metafunc.parametrize(
                argnames, [[funcargs[name] for name in argnames] for funcargs in funcarglist]
            )


class BaseTestTransformer:
    transformer = None

    params = dict(
        test_inversion=[
            dict(r=r)
            for r in np.linspace(0, np.pi / 2, 100)
        ],
    )

    def test_inversion(self, r):
        assert self.transformer.invert(self.transformer(r)) == pytest.approx(r, abs=1e-9)


@pytest.fixture
def lin():
    return LinearTransformer(1.25)


@pytest.fixture
def vsd():
    return ExponentialTransformer(0.95, 0.28, 0.05)


@pytest.fixture
def vsdpq():
    return BiexponentialTransformer(1.2, 0.3, -0.05, 0.2, 0.01)


class TestLinearTransformer(BaseTestTransformer):
    transformer = LinearTransformer(1.25)

    def test_inverse(self, lin):
        assert lin.invert(lin(4.05)) == pytest.approx(4.05, rel=1e-12)


class TestExponentialTransformer(BaseTestTransformer):
    transformer = ExponentialTransformer(0.95, 0.28, 0.05)

    def test_inverse_1(self, vsd):
        assert vsd.invert(vsd(0.775)) == pytest.approx(0.775, rel=1e-12)

    def test_inverse_2(self, vsd):
        assert vsd.invert(vsd(0.27)) == pytest.approx(0.27, rel=1e-12)

    def test_inverse_3(self, vsd):
        assert vsd.invert(vsd(0.999)) == pytest.approx(0.999, rel=1e-12)


class TestBiexpTransformer(BaseTestTransformer):
    transformer = BiexponentialTransformer(1.2, 0.3, -0.05, 0.2, 0.01)

    def test_inverse_1(self, vsdpq):
        assert vsdpq.invert(vsdpq(0.775)) == pytest.approx(0.775, rel=1e-12)

    def test_inverse_2(self, vsdpq):
        assert vsdpq.invert(vsdpq(0.27)) == pytest.approx(0.27, rel=1e-12)

    def test_inverse_3(self, vsdpq):
        assert vsdpq.invert(vsdpq(0.999)) == pytest.approx(0.999, rel=1e-12)

    def test_manual(self, vsdpq):
        assert vsdpq(0.895) == pytest.approx(1.0624794369015114, rel=1e-12)

    def test_manual_inverse(self, vsdpq):
        assert vsdpq.invert(1.0624794369) == pytest.approx(0.895, rel=1e-9)


class TestSaneExponentialTransformer(BaseTestTransformer):
    transformer = SaneExponentialTransformer(0.15, 1, 3.24)


class TestSaneBiexponentialTransformer1(BaseTestTransformer):
    transformer = SaneBiexponentialTransformer(1.1, 0.2, 8.15, 0.3, 2.65)


class TestSaneBiexponentialTransformer2(BaseTestTransformer):
    transformer = SaneBiexponentialTransformer(0.125, 0.12, 1.25, 0.008, 1.35)
