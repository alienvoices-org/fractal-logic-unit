"""Tests for flu.utils.math_helpers — pure arithmetic layer."""

import math
import pytest
from flu.utils.math_helpers import (
    factorial, inv_mod, is_odd,
    digits_signed, digits_unsigned, mean_of_digits,
)


class TestFactorial:
    def test_known_values(self):
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        assert factorial(10) == math.factorial(10)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            factorial(-1)

    def test_cache_consistency(self):
        # Calling twice must return the same value
        assert factorial(7) == factorial(7)


class TestInvMod:
    def test_basic(self):
        assert (3 * inv_mod(3, 7)) % 7 == 1
        assert (5 * inv_mod(5, 11)) % 11 == 1

    def test_self_inverse(self):
        # n-1 is self-inverse mod n when n is prime
        n = 7
        assert (6 * inv_mod(6, n)) % n == 1

    def test_no_inverse_raises(self):
        with pytest.raises(ValueError):
            inv_mod(2, 4)   # gcd(2,4)=2 ≠ 1


class TestIsOdd:
    @pytest.mark.parametrize("n", [1, 3, 5, 7, 9, 11])
    def test_odd(self, n):
        assert is_odd(n) is True

    @pytest.mark.parametrize("n", [2, 4, 6, 8])
    def test_even(self, n):
        assert is_odd(n) is False

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            is_odd(0)


class TestDigitSets:
    @pytest.mark.parametrize("n, expected", [
        (3, [-1, 0, 1]),
        (5, [-2, -1, 0, 1, 2]),
        (7, [-3, -2, -1, 0, 1, 2, 3]),
    ])
    def test_signed_odd(self, n, expected):
        assert digits_signed(n) == expected

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_signed_odd_mean_zero(self, n):
        ds = digits_signed(n)
        assert sum(ds) == 0

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_signed_even_length(self, n):
        ds = digits_signed(n)
        assert len(ds) == n

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_unsigned_range(self, n):
        assert digits_unsigned(n) == list(range(n))

    def test_below_2_raises(self):
        with pytest.raises(ValueError):
            digits_signed(1)
        with pytest.raises(ValueError):
            digits_unsigned(1)


class TestMeanOfDigits:
    @pytest.mark.parametrize("n", [3, 5, 7, 9])
    def test_signed_odd_is_zero(self, n):
        assert mean_of_digits(n, signed=True) == 0.0

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_signed_even_is_half(self, n):
        assert mean_of_digits(n, signed=True) == 0.5

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_unsigned(self, n):
        assert mean_of_digits(n, signed=False) == (n - 1) / 2
