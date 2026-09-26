from copy import deepcopy
import pandas as pd
import json
import pytest
import pymupdf

from research.s20_harness.announcement_adjudicate import decide
from research.s20_harness.announcement_adjudicate import build
from research.s20_harness.announcement_extract import extract_pages
from research.s20_harness.distribution_adapter import row_fingerprint
from research.s20_harness.runtime import atomic_json, digest


def example():
    row = dict(ts_code="600032.SH", stk_div=0., cash_div_tax=.07, record_date="20240704",
               event_terms_usable_for_gross_reference_diagnostic=True,
               conflicting_variants_same_identity=False)
    evidence = dict(code_mentioned=True, cash_status="matches_source",
                    date_roles=dict(all_roles_match_source=True), review_tokens=[],
                    beneficiary_sections=[dict(text='2.分派对象:截至股权登记日下午上海证券交易所收市后,在中国证券登记结算有限责任公司上海分公司(以下简称“中国结算上海分公司”)登记在册的本公司全体股东。')])
    return row, evidence


def test_complete_scope_only_and_no_training_permission():
    row, evidence = example()
    d = decide(row, evidence)
    assert d["accepted_for_retrospective_gross_scope"]
    assert not d["formal_training_eligible"] and not d["historical_availability_proven"]
    for extra in ["但普通流通股不参与。", "（回购专用账户除外）", "其他条件另行公告。"]:
        e = deepcopy(evidence)
        e["beneficiary_sections"][0]["text"] += extra
        assert not decide(row, e)["accepted_for_retrospective_gross_scope"]
    e = deepcopy(evidence)
    e["beneficiary_sections"][0]["text"] = e["beneficiary_sections"][0]["text"].replace("全体股东", "部分股东")
    assert not decide(row, e)["accepted_for_retrospective_gross_scope"]


def test_economic_conflicts_are_not_bypassed_by_good_scope():
    row, e = example()
    for key, value in [("stk_div", .3), ("ts_code", "689009.SH"), ("ts_code", "900901.SH"),
                       ("ts_code", "200001.SZ"), ("ts_code", "999999.SH"), ("cash_div_tax", 0),
                       ("conflicting_variants_same_identity", True)]:
        changed = dict(row, **{key: value})
        assert not decide(changed, e)["accepted_for_retrospective_gross_scope"]
    e["cash_status"] = "multiple_values"
    assert "cash_clause_unresolved" in decide(row, e)["reasons"]
    assert "normalized_terms_unresolved" in decide(dict(row, conflicting_variants_same_identity=pd.NA), e)["reasons"]


def test_explicit_scope_date_must_equal_record_date():
    row, e = example()
    e["beneficiary_sections"][0]["text"] = e["beneficiary_sections"][0]["text"].replace("股权登记日", "2024年7月3日")
    assert not decide(row, e)["accepted_for_retrospective_gross_scope"]


@pytest.mark.parametrize("ending", [
    "(不含本公司回购专用证券账户)。",
    "(本公司回购专用证券账户除外)。",
    "。根据《上海证券交易所上市公司自律监管指引第7号——回购股份》等有关规定,公司存放于回购专用证券账户中的股份不参与本次利润分配。",
])
def test_v2_only_recognizes_complete_issuer_repurchase_exclusions(ending):
    row, e = example()
    e["beneficiary_sections"][0]["text"] = e["beneficiary_sections"][0]["text"][:-1] + ending
    assert decide(row, e)["accepted_for_retrospective_gross_scope"]
    e["beneficiary_sections"][0]["text"] += "普通投资者不参与。"
    assert not decide(row, e)["accepted_for_retrospective_gross_scope"]


def test_other_holder_exclusions_cannot_be_rewritten_as_repurchase():
    row, e = example()
    e["beneficiary_sections"][0]["text"] = e["beneficiary_sections"][0]["text"].replace("全体股东。", "全体股东(不含个人投资者)。")
    assert not decide(row, e)["accepted_for_retrospective_gross_scope"]


@pytest.mark.parametrize("tamper", [None, "pages", "evidence"])
def test_builder_reconstructs_pdf_and_evidence_even_if_cache_rehashed(tmp_path, tamper):
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    source = tmp_path / "normalized.parquet"
    row, _ = example()
    row.update(normalized_event_id="event", ex_date="20240705", pay_date="20240705")
    pd.DataFrame([row]).to_parquet(source)
    frame = pd.read_parquet(source)
    pdf_path = extraction / "document.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "Unrecognized fixture document")
        doc.save(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        pages = [p.get_text() for p in doc]
    url = "finalpage/2024-06-28/123.PDF"
    documents = {url: dict(pdf_path=str(pdf_path), pdf_sha256=digest(pdf_path), pages=pages)}
    events = [dict(normalized_event_id="event", row_sha256=row_fingerprint(frame.iloc[0]),
                   extraction_status="diagnostic_extracted", candidate_paths=[url],
                   evidence=extract_pages(pages, frame.iloc[0]), ts_code=row["ts_code"])]
    if tamper == "pages":
        documents[url]["pages"] = ["Replaced cached document text"]
    elif tamper == "evidence":
        events[0]["evidence"]["cash_status"] = "matches_source"
    atomic_json(extraction / "documents.json", documents)
    atomic_json(extraction / "events.json", events)
    atomic_json(extraction / "inputs.json", [dict(path=str(pdf_path), sha256=digest(pdf_path))])
    atomic_json(extraction / "summary.json", dict(artifacts={n:digest(extraction/n) for n in ["documents.json", "events.json", "inputs.json"]}))
    if tamper:
        with pytest.raises(ValueError, match="reconstruction mismatch|evidence/code mismatch"):
            build(tmp_path, extraction, digest(extraction/"summary.json"), source, digest(source))
        assert not (tmp_path / "output").exists()
    else:
        result = build(tmp_path, extraction, digest(extraction/"summary.json"), source, digest(source))
        assert result["event_rows"] == result["unresolved"] == 1
        assert result["accepted_scope_reviews"] == 0
