import pytest
import math
import numpy as np

from tests.projections.base import pytest_generate_tests, BaseTestProjection
from demeteor.projections.zenith import ZenithShifter


@pytest.fixture
def zenith_aligned():
    return ZenithShifter(0, 0)


@pytest.fixture
def general():
    return ZenithShifter(math.radians(1.5), math.radians(213.4))


@pytest.fixture
def general_2():
    return ZenithShifter(math.radians(17.3), math.radians(107.4))


class BaseTestZenithShifter:
    grid = [
        dict(r=r, t=t)
        for r in np.linspace(0.01, math.tau / 4, 11)
        for t in np.linspace(0.05, math.tau + 0.05, 11, endpoint=False)
    ]
    params = dict(
        test_zenith_aligned=grid,
    )

    shifter = None

    def test_zenith_aligned(self, r, t, request):
        assert self.shifter.invert(*self.shifter(r, t)) == pytest.approx((r, t), abs=1e-9)


class TestZenithShifter1(BaseTestZenithShifter):
    shifter = ZenithShifter(math.radians(1.5), math.radians(213.4))


class TestZenithShifter2(BaseTestZenithShifter):
    shifter = ZenithShifter(math.radians(28.5), math.radians(113.4))


class TestZenithShifter3(BaseTestZenithShifter):
    shifter = ZenithShifter(math.radians(0.01), 0)


class TestZenithShifter4(BaseTestZenithShifter):
    shifter = ZenithShifter(math.radians(-45.01), math.radians(23.56))


class TestZenithShifterZero(BaseTestZenithShifter):
    shifter = ZenithShifter(math.radians(0), math.radians(4.5))


