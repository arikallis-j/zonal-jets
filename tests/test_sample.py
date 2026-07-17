import pytest

def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5

def test_sum():
    assert (0.1 + 0.2) == pytest.approx(0.3)