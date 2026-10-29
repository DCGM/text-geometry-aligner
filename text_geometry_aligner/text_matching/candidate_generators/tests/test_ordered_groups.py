"""Ordered record anchors without imposing a sequence on their fields."""
from pathlib import Path

import pytest
import text_geometry_aligner as alignment


def _generate(data, words, normalizer):
    regions = alignment.JSONTextReader().from_data(data).regions
    for region in regions:
        region.input_text_normalized = normalizer.normalize(str(region.input_text))
    page = alignment.ALTOPage(source_path=Path('test.xml'), words=tuple(
        alignment.ALTOWord(index=i, text=text, bbox=alignment.BoundingBox(x, y, 9, 10))
        for i, (text, x, y) in enumerate(words)))
    generator = alignment.OrderedGroupsCandidateGenerator(alignment.ExactTextCandidateGenerator())
    return generator.generate(regions, alignment.ALTOTextIndex(page, normalizer))


def test_number_before_title_does_not_follow_json_field_order(lowercase_normalizer):
    candidates = _generate({'entries': [{'title': 'Alpha', 'page': '1'},
                                         {'title': 'Beta', 'page': '2'}]},
        [('1', 100, 0), ('Alpha', 0, 0), ('2', 100, 20), ('Beta', 0, 20)], lowercase_normalizer)
    assert {(c.json_text_path[1:], c.start_word) for c in candidates} == {
        ((0, 'title'), 1), ((0, 'page'), 0), ((1, 'title'), 3), ((1, 'page'), 2)}


def test_separate_number_ocr_block_does_not_require_entry_intervals(lowercase_normalizer):
    candidates = _generate({'entries': [{'title': 'Alpha', 'page': '1'},
                                         {'title': 'Beta', 'page': '2'}]},
        [('Alpha', 0, 0), ('Beta', 0, 20), ('1', 100, 0), ('2', 100, 20)], lowercase_normalizer)
    assert len(candidates) == 4
    assert {c.start_word for c in candidates if c.json_text_path[1] == 0} == {0, 2}
    assert {c.start_word for c in candidates if c.json_text_path[1] == 1} == {1, 3}


def test_repeated_titles_follow_entry_order_and_fields_use_local_context(lowercase_normalizer):
    candidates = _generate({'entries': [{'title': 'Repeat', 'page': '7'},
                                         {'title': 'Repeat', 'page': '7'}]},
        [('Repeat', 0, 0), ('7', 100, 0), ('Repeat', 0, 20), ('7', 100, 20)], lowercase_normalizer)
    assert {c.start_word for c in candidates if c.json_text_path[1] == 0} == {0, 1}
    assert {c.start_word for c in candidates if c.json_text_path[1] == 1} == {2, 3}


def test_unmatched_entry_does_not_prevent_later_anchors(lowercase_normalizer):
    candidates = _generate({'entries': [{'title': 'Alpha'}, {'title': 'Missing'}, {'title': 'Beta'}]},
        [('Alpha', 0, 0), ('Beta', 0, 20)], lowercase_normalizer)
    assert [(c.json_text_path[1], c.start_word) for c in candidates] == [(0, 0), (2, 1)]


@pytest.mark.parametrize('depth', [0, -1, True, 1.5])
def test_group_depth_must_be_positive_integer(depth):
    with pytest.raises(ValueError, match='positive integer'):
        alignment.OrderedGroupsCandidateGenerator(alignment.ExactTextCandidateGenerator(), group_depth=depth)


def test_cli_builds_the_optional_generator():
    from text_geometry_aligner.text_aligner import build_argument_parser, _build_candidate_generator
    args = build_argument_parser().parse_args(['--input-dir', 'in', '--alto-dir', 'alto',
        '--json-output-dir', 'out', '--candidate-generator', 'ordered-groups', '--ordered-group-depth', '2'])
    assert isinstance(_build_candidate_generator(args), alignment.OrderedGroupsCandidateGenerator)
