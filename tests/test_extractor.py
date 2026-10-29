"""Offline tests: no API key or network needed."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rfp_extractor as rx  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def test_parse_model_json_plain():
    assert rx.parse_model_json('{"Title": "X"}') == {"Title": "X"}


def test_parse_model_json_fenced_with_chatter():
    text = 'Here you go:\n```json\n{"Title": "X"}\n```\nDone.'
    assert rx.parse_model_json(text) == {"Title": "X"}


def test_parse_model_json_surrounding_text():
    assert rx.parse_model_json('Sure! {"a": 1} hope that helps') == {"a": 1}


def test_from_dict_normalises_schema():
    rec = rx.RFPData.from_dict({
        "Title": "  Laptops ", "Model_no": ["A1", "B2"],
        "Due_Date": None, "unexpected": "dropped",
    })
    out = rec.__dict__
    assert set(out) == set(rx.RFPData.field_names())
    assert out["Title"] == "Laptops"
    assert out["Model_no"] == "A1; B2"
    assert out["Due_Date"] == ""
    assert len(out) == 21


def test_chunk_text_overlap():
    words = " ".join(str(i) for i in range(1000))
    chunks = rx.chunk_text(words, size=400, overlap=80)
    assert chunks[0].split()[-80:] == chunks[1].split()[:80]
    assert chunks[-1].split()[-1] == "999"


def test_chunk_text_rejects_bad_params():
    with pytest.raises(ValueError):
        rx.chunk_text("a b c", size=10, overlap=10)


def test_safe_filename():
    assert rx.safe_filename("Bid1/Addendum 1 RFP.pdf") == "Bid1_Addendum_1_RFP.json"


@pytest.mark.skipif(not DATA.exists(), reason="sample data not present")
def test_collect_documents_from_sample_dirs():
    docs = rx.collect_documents([str(DATA / "Bid1"), str(DATA / "Bid2")])
    keys = [k for _, k in docs]
    assert len(docs) == 9
    assert all(k.startswith(("Bid1/", "Bid2/")) for k in keys)


@pytest.mark.skipif(not DATA.exists(), reason="sample data not present")
def test_parsers_extract_text():
    parser = rx.DocumentParser()
    pdf = next((DATA / "Bid2").glob("*.pdf"))
    html = next((DATA / "Bid2").glob("*.html"))
    assert len(parser.parse(pdf)) > 500
    assert len(parser.parse(html)) > 500
