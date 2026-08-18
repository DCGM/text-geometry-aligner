"""Tests for the top-level package's public API surface."""

import pytest

import text_geometry_aligner as alignment


def test_direction_specific_aligners_replace_old_orchestrator() -> None:
    assert alignment.TextAligner.__module__ == "text_geometry_aligner.text_aligner"
    assert (
        alignment.GeometryAligner.__module__
        == "text_geometry_aligner.geometry_aligner"
    )
    assert not hasattr(alignment, "TextGeometryAligner")
    assert not hasattr(alignment, "AlignmentDirection")


def test_aligner_requires_generator_and_selector() -> None:
    with pytest.raises(TypeError):
        alignment.TextAligner()
