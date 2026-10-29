"""Clear gutters, preserved geometry, and isolated stitched source pages."""
from dataclasses import replace
from pathlib import Path

import pytest
from text_geometry_aligner import (
    ALTOPage, ALTOWord, BoundingBox, ColumnReadingOrder,
    CPSATCandidateSelector, ExactTextCandidateGenerator,
    OrderedGroupsCandidateGenerator, TextAligner,
)


def _page(y_offset=0):
    words = []
    for row in range(6):
        for column, prefix in enumerate(['Applesubject', 'Zebrathing']):
            i = len(words)
            words.append(ALTOWord(index=i, text=f'{prefix}{row}',
                bbox=BoundingBox(30 + 260 * column, y_offset + 20 * row + 20, 120, 12),
                line_index=i, block_index=column, alto_line_id=f'line{i}'))
    return ALTOPage(source_path=Path('test.xml'), words=tuple(words), width=500, height=180)


def test_interleaved_rows_become_left_column_then_right():
    page = _page()
    order = ColumnReadingOrder()
    result = order.reorder(page)
    assert [w.text for w in result.words] == [w.text for w in page.words[::2] + page.words[1::2]]
    assert len(order.detected_regions) == 1
    assert page.words[1].text == 'Zebrathing0'  # Original is untouched.
    originals = {w.alto_line_id: w for w in page.words}
    for i, word in enumerate(result.words):
        assert word.index == i
        assert replace(word, index=originals[word.alto_line_id].index) == originals[word.alto_line_id]


def test_full_width_heading_stays_before_both_columns():
    page = _page()
    heading = ALTOWord(index=12, text='Heading', bbox=BoundingBox(0, 0, 500, 12), line_index=12)
    result = ColumnReadingOrder().reorder(replace(page, words=page.words + (heading,)))
    assert result.words[0].text == 'Heading'
    assert result.words[-1].text == 'Zebrathing5'


def test_spanning_middle_heading_separates_column_bands():
    first = _page()
    lower = tuple(replace(w, index=w.index + 12, line_index=w.line_index + 12,
                          bbox=replace(w.bbox, y=w.bbox.y + 200), text='Lower' + w.text)
                  for w in first.words)
    heading = ALTOWord(index=24, text='Section', bbox=BoundingBox(0, 170, 500, 12), line_index=24)
    result = ColumnReadingOrder().reorder(replace(first, words=first.words + lower + (heading,), height=400))
    assert result.words[12].text == 'Section'
    assert all(w.bbox.y < 170 for w in result.words[:12])
    assert all(w.bbox.y > 170 for w in result.words[13:])


def test_existing_column_order_is_preserved_exactly():
    page = _page()
    words = page.words[::2] + page.words[1::2]
    page = replace(page, words=tuple(replace(w, index=i) for i, w in enumerate(words)))
    order = ColumnReadingOrder()
    assert order.reorder(page) is page
    assert len(order.detected_regions) == 1
    assert not order.detected_regions[0]['reordered']


def test_stitched_source_pages_do_not_mix_their_columns():
    first = _page()
    second = _page(y_offset=180)
    second_words = tuple(replace(w, index=w.index + 12, line_index=w.line_index + 12,
                                 alto_line_id='second_' + w.alto_line_id) for w in second.words)
    page = replace(first, words=first.words + second_words, height=360)
    order = ColumnReadingOrder([BoundingBox(0, 0, 500, 180), BoundingBox(0, 180, 500, 180)])
    result = order.reorder(page)
    assert len(order.detected_regions) == 2
    assert all(w.bbox.y < 180 for w in result.words[:12])
    assert all(w.bbox.y >= 180 for w in result.words[12:])


@pytest.mark.parametrize('number', ['7', 'XVIII'])
def test_number_column_alone_does_not_make_a_second_text_column(number):
    page = _page()
    words = tuple(replace(w, text=number) if i % 2 else w for i, w in enumerate(page.words))
    page = replace(page, words=words)
    assert ColumnReadingOrder().reorder(page) is page


def test_vertically_separated_text_is_not_two_parallel_columns():
    page = _page()
    page = replace(page, words=tuple(replace(w, bbox=replace(w.bbox, y=w.bbox.y + 300))
                                    if i % 2 else w for i, w in enumerate(page.words)), height=500)
    assert ColumnReadingOrder().reorder(page) is page


def test_ordered_group_anchors_use_the_reordered_view(lowercase_normalizer):
    page = _page()
    data = {'entries': [{'title': w.text} for w in page.words[::2] + page.words[1::2]]}
    aligner = TextAligner(candidate_generator=OrderedGroupsCandidateGenerator(
        ExactTextCandidateGenerator(), reading_order=ColumnReadingOrder()),
        candidate_selector=CPSATCandidateSelector(), normalizer=lowercase_normalizer)
    result = aligner.align_data(page, data).pages[0]
    assert result.matched_count == 12
    assert all(r.alto_text == r.input_text for r in result.regions)
    assert all(r.alto_geometry.x == 30 for r in result.regions[:6])
    assert all(r.alto_geometry.x == 290 for r in result.regions[6:])


def test_invalid_reading_region():
    with pytest.raises(ValueError, match='positive dimensions'):
        ColumnReadingOrder([BoundingBox(0, 0, 0, 100)])


def test_column_cli_option_is_explicit_and_scoped_to_ordered_groups():
    from text_geometry_aligner.text_aligner import build_argument_parser, _build_candidate_generator
    base = ['--input-dir', 'in', '--alto-dir', 'alto', '--json-output-dir', 'out']
    args = build_argument_parser().parse_args(base + ['--candidate-generator', 'ordered-groups', '--ordered-group-columns'])
    assert isinstance(_build_candidate_generator(args).reading_order, ColumnReadingOrder)
    args = build_argument_parser().parse_args(base + ['--ordered-group-columns'])
    with pytest.raises(ValueError, match='requires'):
        _build_candidate_generator(args)
