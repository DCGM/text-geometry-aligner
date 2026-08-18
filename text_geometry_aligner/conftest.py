import pytest

from text_geometry_aligner.label_mapping import LabelMapper
from text_geometry_aligner.normalization import TextNormalizationPipeline


@pytest.fixture
def alignment_output():
    return lambda aligner, document: aligner.json_writer.to_data(
        document.pages[0]
    )


@pytest.fixture
def lowercase_normalizer() -> TextNormalizationPipeline:
    return TextNormalizationPipeline.from_optional_names(("lowercase",))


@pytest.fixture
def label_mapper() -> LabelMapper:
    return LabelMapper.from_data(
        {
            "Title": "heading",
            "YOLO title": "heading",
        }
    )
