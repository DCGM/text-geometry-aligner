"""Tests for the ALTO XML reader."""

import text_geometry_aligner as alignment


def test_alto_reader_only_parses_xml_into_page_model(tmp_path) -> None:
    alto_xml = """\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout>
    <Page ID="page-1" WIDTH="100" HEIGHT="200">
      <TextBlock>
        <TextLine>
          <String ID="word-1" CONTENT="ROME"
                  HPOS="1" VPOS="2" WIDTH="30" HEIGHT="10"/>
        </TextLine>
      </TextBlock>
    </Page>
  </Layout>
</alto>
"""
    input_path = tmp_path / "page.xml"
    input_path.write_text(alto_xml, encoding="utf-8")

    page = alignment.ALTOReader().read(input_path)

    assert page.page_id == "page-1"
    assert (page.width, page.height) == (100.0, 200.0)
    assert len(page.words) == 1
    assert page.words[0].text == "ROME"
    assert page.words[0].alto_word_id == "word-1"
    assert page.words[0].alto_line_id is None
    assert page.words[0].bbox == alignment.BoundingBox(1.0, 2.0, 30.0, 10.0)


def test_alto_reader_keeps_line_and_word_ids_only_where_the_alto_has_them(tmp_path) -> None:
    alto_xml = """\
<alto xmlns="http://www.loc.gov/standards/alto/ns-v2#">
  <Layout>
    <Page WIDTH="100" HEIGHT="200">
      <TextBlock>
        <TextLine ID="line-1">
          <String ID="word-1" CONTENT="ROMA"
                  HPOS="1" VPOS="2" WIDTH="30" HEIGHT="10"/>
          <String CONTENT="AETERNA"
                  HPOS="35" VPOS="2" WIDTH="40" HEIGHT="10"/>
        </TextLine>
        <TextLine>
          <String CONTENT="MMXXVI"
                  HPOS="1" VPOS="20" WIDTH="30" HEIGHT="10"/>
        </TextLine>
      </TextBlock>
    </Page>
  </Layout>
</alto>
"""
    input_path = tmp_path / "page.xml"
    input_path.write_text(alto_xml, encoding="utf-8")

    page = alignment.ALTOReader().read(input_path)

    assert [(word.alto_word_id, word.alto_line_id) for word in page.words] == [
        ("word-1", "line-1"),
        (None, "line-1"),
        (None, None),
    ]
