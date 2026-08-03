"""Tests for real storage specs parsed from FDA drug labels."""

from __future__ import annotations

from pathlib import Path

import pytest

from coldspend.validate import parse_allowance, parse_range, summarise
from coldspend.validate.labels import CACHE, fetch_labels

needs_cache = pytest.mark.skipif(not CACHE.exists(), reason="label cache absent")


def test_degree_symbol_variants_all_parse():
    """THE trap, and how it was found: a regex expecting a plain degree sign
    matched ZERO of 500 labels. The corpus uses U+00B0, U+00BA, a superscript
    zero, a bare 'o', and sometimes nothing."""
    for deg in ("\u00b0", "\u00ba", "\u2070", "o", ""):
        assert parse_range(f"Store at 2{deg}C to 8{deg}C") == (2, 8), f"failed on {deg!r}"


def test_excursion_allowance_is_extracted():
    """The phrase the GMP claim rests on, in the regulator's own words."""
    t = "Store at 25 C; excursions permitted between 15 - 30 C (59-86 F)"
    assert parse_allowance(t) == (15, 30)


def test_allowance_requires_the_excursion_wording():
    """A plain storage range is not an allowance. Conflating them would inflate
    the headroom this project claims exists."""
    assert parse_allowance("Store at 20 C to 25 C") is None


def test_implausible_ranges_are_rejected():
    assert parse_range("call 1-800 between 9 to 5 C") != (1, 800)
    assert parse_range("store at 200 C to 900 C") is None


@needs_cache
def test_real_corpus_yields_ranges_and_allowances():
    s = summarise(fetch_labels())
    assert s.n_ranges > 300, "parser regression: real labels do state ranges"
    assert s.n_with_allowance > 50
    assert 0.0 < s.mean_headroom_c < 15.0


@needs_cache
def test_cold_chain_is_a_minority_of_labels():
    """Worth knowing and mildly counter-intuitive: 2-8 C is a few percent of
    drug labels. Cold chain matters for what is in it, not its share."""
    s = summarise(fetch_labels())
    assert s.refrigerated_share < 0.15
