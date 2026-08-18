"""Tests for the shared file/directory processing base."""

import json
from pathlib import Path
from unittest import mock

import pytest

import text_geometry_aligner as alignment


def _page(*texts: str, block_indexes: tuple[int, ...] | None = None) -> alignment.ALTOPage:
    if block_indexes is None:
        block_indexes = (0,) * len(texts)
    words = tuple(
        alignment.ALTOWord(
            index=index,
            text=text,
            bbox=alignment.BoundingBox(index * 10, 0, 9, 10),
            line_index=0,
            block_index=block_indexes[index],
        )
        for index, text in enumerate(texts)
    )
    return alignment.ALTOPage(source_path=Path("test.xml"), words=words)


class _FirstCandidateSelector:
    def select(self, candidates, regions):
        return tuple(candidates[:1])


def _alto_xml(word: str) -> str:
    return f"""\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout>
    <Page ID="page-id" WIDTH="100" HEIGHT="200">
      <TextBlock>
        <TextLine>
          <String ID="word-id" CONTENT="{word}"
                  HPOS="10" VPOS="20" WIDTH="30" HEIGHT="10"/>
        </TextLine>
      </TextBlock>
    </Page>
  </Layout>
</alto>
"""


def test_process_file_delegates_format_io() -> None:
    alto_reader = mock.create_autospec(
        alignment.ALTOReader,
        instance=True,
    )
    json_reader = mock.create_autospec(
        alignment.JSONTextReader,
        instance=True,
    )
    json_writer = mock.create_autospec(
        alignment.AlignmentJSONWriter,
        instance=True,
    )
    alto_reader.read.return_value = _page("ROME")
    json_reader.read.return_value = alignment.JSONTextReader().from_data(
        {"title": "Rome"},
        page_key="input",
        input_file_path=Path("input.json"),
    )
    aligner = alignment.TextAligner(
        alto_reader=alto_reader,
        json_reader=json_reader,
        json_writer=json_writer,
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_FirstCandidateSelector(),
    )

    result = aligner.process_file(
        "page.xml",
        "input.json",
        "output.json",
    )

    alto_reader.read.assert_called_once_with(Path("page.xml"))
    json_reader.read.assert_called_once_with(
        Path("input.json"),
        page_key="input",
    )
    json_writer.write.assert_called_once_with(
        result.pages[0],
        Path("output.json"),
    )


def test_process_file_can_return_document_without_export() -> None:
    alto_reader = mock.create_autospec(
        alignment.ALTOReader,
        instance=True,
    )
    json_reader = mock.create_autospec(
        alignment.JSONTextReader,
        instance=True,
    )
    json_writer = mock.create_autospec(
        alignment.AlignmentJSONWriter,
        instance=True,
    )
    alto_reader.read.return_value = _page("ROME")
    json_reader.read.return_value = alignment.JSONTextReader().from_data(
        {"title": "Rome"},
        page_key="input",
        input_file_path=Path("input.json"),
    )
    aligner = alignment.TextAligner(
        alto_reader=alto_reader,
        json_reader=json_reader,
        json_writer=json_writer,
        candidate_generator=alignment.ExactTextCandidateGenerator(),
        candidate_selector=_FirstCandidateSelector(),
    )

    result = aligner.process_file(
        alto_file="page.xml",
        input_file="input.json",
    )

    json_writer.write.assert_not_called()
    assert len(result.pages) == 1
    assert result.pages[0].regions[0].input_text == "Rome"


def test_process_files_pairs_by_key_and_preserves_input_order(tmp_path) -> None:
    root = tmp_path
    input_dir = root / "input"
    alto_dir = root / "alto"
    input_dir.mkdir()
    alto_dir.mkdir()
    first_input = input_dir / "first"
    second_input = input_dir / "second.labels"
    first_alto = alto_dir / "first.xml"
    second_alto = alto_dir / "second.xml"
    first_input.write_text(
        "0 25 25 30 10 0.8 PageNumber\n",
        encoding="utf-8",
    )
    second_input.write_text(
        "0 25 25 30 10 0.9 PageNumber\n",
        encoding="utf-8",
    )
    first_alto.write_text(_alto_xml("1"), encoding="utf-8")
    second_alto.write_text(_alto_xml("2"), encoding="utf-8")
    json_writer = mock.create_autospec(
        alignment.AlignmentJSONWriter,
        instance=True,
    )

    document = alignment.GeometryAligner(
        json_writer=json_writer,
    ).process_files(
        alto_files=[first_alto, second_alto],
        input_files=[second_input, first_input],
        input_format=alignment.InputFormat.YOLO,
    )

    json_writer.write.assert_not_called()
    assert [page.page_key for page in document.pages] == ["second", "first"]
    assert [page.regions[0].alto_text for page in document.pages] == [
        "2",
        "1",
    ]
    assert document.input_path == input_dir
    assert document.alto_path == alto_dir
    assert document.pages[0].input_file_path == second_input
    assert document.pages[0].alto_file_path == second_alto


def test_process_files_supports_json_input_and_export(tmp_path) -> None:
    root = tmp_path
    input_file = root / "page.json"
    alto_file = root / "page.xml"
    output_dir = root / "output"
    input_file.write_text(
        json.dumps(
            {
                "PageNumber_bbox": {
                    "x": 10,
                    "y": 20,
                    "width": 30,
                    "height": 10,
                }
            }
        ),
        encoding="utf-8",
    )
    alto_file.write_text(_alto_xml("12"), encoding="utf-8")

    document = alignment.GeometryAligner().process_files(
        alto_files=[alto_file],
        input_files=[input_file],
        json_output_dir=output_dir,
    )

    assert document.pages[0].regions[0].alto_text == "12"
    assert (
        json.loads((output_dir / "page.json").read_text(encoding="utf-8"))[
            "PageNumber"
        ]
        == "12"
    )


def test_process_files_validates_page_keys_and_missing_alto(tmp_path) -> None:
    root = tmp_path
    first_input = root / "page.labels"
    duplicate_input = root / "page.txt"
    first_input.write_text("", encoding="utf-8")
    duplicate_input.write_text("", encoding="utf-8")

    aligner = alignment.GeometryAligner()
    with pytest.raises(ValueError, match="Multiple yolo input"):
        aligner.process_files(
            alto_files=[],
            input_files=[first_input, duplicate_input],
            input_format=alignment.InputFormat.YOLO,
        )

    document = aligner.process_files(
        alto_files=[],
        input_files=[first_input],
        input_format=alignment.InputFormat.YOLO,
    )
    assert document.pages == []

    with pytest.raises(FileNotFoundError, match="No ALTO XML"):
        aligner.process_files(
            alto_files=[],
            input_files=[first_input],
            input_format=alignment.InputFormat.YOLO,
            fail_on_missing_alto=True,
        )


def test_directory_pairing_removes_one_suffix_or_uses_full_name(tmp_path) -> None:
    root = tmp_path
    input_dir = root / "input"
    alto_dir = root / "alto"
    output_dir = root / "output"
    input_dir.mkdir()
    alto_dir.mkdir()

    (input_dir / "page.full.labels").write_text(
        "0 25 25 30 10 0.9 PageNumber\n",
        encoding="utf-8",
    )
    (alto_dir / "page.full.xml").write_text(
        _alto_xml("12"),
        encoding="utf-8",
    )
    (input_dir / "plain").write_text(
        "0 25 25 30 10 0.8 PageNumber\n",
        encoding="utf-8",
    )
    (alto_dir / "plain.xml").write_text(
        _alto_xml("13"),
        encoding="utf-8",
    )

    document = alignment.GeometryAligner().process_directories(
        alto_dir,
        input_dir,
        output_dir,
        input_format=alignment.InputFormat.YOLO,
    )

    assert [page.page_key for page in document.pages] == ["page.full", "plain"]
    assert document.input_path == input_dir
    assert document.alto_path == alto_dir
    assert document.matched_count == 2
    assert document.pages[0].json_source_data is None
    assert document.pages[0].alto_width == 100
    assert document.pages[0].alto_height == 200
    assert (
        json.loads(
            (output_dir / "page.full.json").read_text(encoding="utf-8")
        )["PageNumber"]
        == ["12"]
    )


def test_directory_processing_can_return_document_without_export(tmp_path) -> None:
    root = tmp_path
    input_dir = root / "input"
    alto_dir = root / "alto"
    input_dir.mkdir()
    alto_dir.mkdir()
    (input_dir / "page.labels").write_text(
        "0 25 25 30 10 0.9 PageNumber\n",
        encoding="utf-8",
    )
    (alto_dir / "page.xml").write_text(
        _alto_xml("12"),
        encoding="utf-8",
    )
    json_writer = mock.create_autospec(
        alignment.AlignmentJSONWriter,
        instance=True,
    )

    document = alignment.GeometryAligner(
        json_writer=json_writer,
    ).process_directories(
        alto_dir,
        input_dir,
        input_format=alignment.InputFormat.YOLO,
    )

    json_writer.write.assert_not_called()
    assert len(document.pages) == 1
    assert document.pages[0].page_key == "page"
    assert document.pages[0].regions[0].alto_text == "12"
    assert document.matched_count == 1
