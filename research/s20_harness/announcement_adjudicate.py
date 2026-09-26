"""Source-bound complete-template adjudication for retrospective cash accounting.

Not a general Chinese-language semantic classifier. Every unsupported section,
qualification or economic unit stays unresolved; no PIT/training approval.
"""
from collections import Counter
import json
from pathlib import Path
import re
import uuid

import pandas as pd
import pymupdf

from .announcement_extract import extract_pages
from .distribution_adapter import row_fingerprint
from .runtime import atomic_json, digest, now


POLICY = "ordinary_cash_complete_scope_template_v2"


def decide(row, evidence):
    reasons = []
    # Fail closed for B shares, CDRs and unregistered/new code families.
    # Exchange suffix alone does not establish an ordinary A-share cash unit.
    if not re.fullmatch(r"(?:(?:600|601|603|605|688)\d{3}\.SH|(?:000|001|002|003|300|301)\d{3}\.SZ)", str(row["ts_code"])):
        reasons.append("unsupported_market_or_unit")
    if pd.isna(row["stk_div"]) or float(row["stk_div"]) != 0:
        reasons.append("share_terms_require_review")
    if pd.isna(row["cash_div_tax"]) or float(row["cash_div_tax"]) <= 0:
        reasons.append("positive_cash_required")
    usable, conflict = row["event_terms_usable_for_gross_reference_diagnostic"], row["conflicting_variants_same_identity"]
    if pd.isna(usable) or usable != True or pd.isna(conflict) or bool(conflict):
        reasons.append("normalized_terms_unresolved")
    if not evidence["code_mentioned"]:
        reasons.append("security_code_not_confirmed")
    if evidence["cash_status"] != "matches_source":
        reasons.append("cash_clause_unresolved")
    if not evidence["date_roles"]["all_roles_match_source"]:
        reasons.append("date_roles_unresolved")
    if any(r["token"] in ("重整", "债权人", "存托凭证") for r in evidence["review_tokens"]):
        reasons.append("special_economic_context")
    sections = evidence["beneficiary_sections"]
    if len(sections) != 1:
        reasons.append("unique_complete_scope_required")
    else:
        scope = sections[0]["text"]
        heading = r"(?:2[.、]分派对象:|\(二\)分派对象:|[一二三四五六七八九十]+[、.](?:权益分派对象|分红派息对象):?(?:本次(?:权益)?分派对象为:)?)"
        date = str(row["record_date"])
        # Exact record date in prose, or explicit named record day; not any
        # date elsewhere in the document. No wildcard for holder exclusions.
        dated = f"{date[:4]}年{int(date[4:6])}月{int(date[6:])}日" if re.fullmatch(r"\d{8}", date) else "NEVER_MATCH"
        market = "上海" if str(row["ts_code"]).endswith(".SH") else "深圳"
        registrar = "中国证券登记结算有限责任公司" + market + "分公司"
        short = "中国结算" + market + "分公司"
        alias = r"\((?:以下简称|简称)[“\"]?" + short + r"[”\"]?\)"
        registrar_pattern = "(?:" + registrar + "(?:" + alias + ")?|" + short + ")"
        # V2 recognizes only explicitly issuer-owned repurchase exclusions.
        # Never strip generic exclusions, names, footers or extra qualifications.
        holder = (r"登记在册的(?:除公司回购专用证券账户外的)?(?:本公司|公司)全体股东"
                  r"(?:\((?:不含本公司回购专用证券账户|本公司回购专用证券账户除外)\))?。")
        tail = (r"(?:根据《上海证券交易所上市公司自律监管指引第7号——回购股份》等有关规定,"
                r"公司存放于回购专用证券账户(?:中)?的股份不参与(?:本次)?利润分配。)?")
        pattern = (heading + r"(?:截至|截止)(?:股权登记日|" + dated + r")下午" + market
                   + r"证券交易所收市后,在" + registrar_pattern + holder + tail)
        if not re.fullmatch(pattern, scope):
            reasons.append("scope_outside_complete_template")
    return dict(policy_id=POLICY, accepted_for_retrospective_gross_scope=not reasons,
                reasons=reasons, formal_training_eligible=False, historical_availability_proven=False)


def build(root, extraction, extraction_sha, normalized_path, normalized_sha):
    extraction, normalized_path = Path(extraction).resolve(), Path(normalized_path).resolve()
    pins = [(extraction / "summary.json", extraction_sha), (normalized_path, normalized_sha)]
    def verify():
        if any(digest(p) != sha for p, sha in pins):
            raise ValueError("adjudication input pin mismatch")
    verify()
    summary = json.loads((extraction / "summary.json").read_text(encoding="utf-8"))
    pins.extend((extraction / name, sha) for name, sha in summary["artifacts"].items())
    verify()
    events = json.loads((extraction / "events.json").read_text(encoding="utf-8"))
    docs = json.loads((extraction / "documents.json").read_text(encoding="utf-8"))
    inputs = json.loads((extraction / "inputs.json").read_text(encoding="utf-8"))
    pins.extend((Path(i["path"]), i["sha256"]) for i in inputs)
    verify()
    frame = pd.read_parquet(normalized_path).set_index("normalized_event_id", drop=False)
    if not frame.index.is_unique or len({e["normalized_event_id"] for e in events}) != len(events):
        raise ValueError("duplicate event identities")
    # Reconstruct page texts from retained PDF bytes rather than trusting
    # editable extracted evidence or a cached approval flag.
    for doc in docs.values():
        p = Path(doc["pdf_path"])
        if (p, doc["pdf_sha256"]) not in pins or digest(p) != doc["pdf_sha256"]:
            raise ValueError("PDF not bound to extraction inputs")
        with pymupdf.open(p) as pdf:
            if [page.get_text() for page in pdf] != doc["pages"]:
                raise ValueError("PDF text reconstruction mismatch")
    reviews, decisions = {}, []
    for e in events:
        row = frame.loc[e["normalized_event_id"]]
        if row_fingerprint(row) != e["row_sha256"]:
            raise ValueError("event row changed")
        if e["extraction_status"] != "diagnostic_extracted":
            decision = dict(policy_id=POLICY, accepted_for_retrospective_gross_scope=False,
                            reasons=["document_unresolved"], formal_training_eligible=False,
                            historical_availability_proven=False)
        else:
            doc = docs[e["candidate_paths"][0]]
            evidence = extract_pages(doc["pages"], row)
            if evidence != e["evidence"]:
                raise ValueError("extraction evidence/code mismatch")
            decision = decide(row, evidence)
            if decision["accepted_for_retrospective_gross_scope"]:
                reviews[e["normalized_event_id"]] = dict(
                    row_sha256=e["row_sha256"], ts_code=e["ts_code"],
                    beneficiary_scope="existing_shareholders_verified",
                    source="https://static.cninfo.com.cn/" + e["candidate_paths"][0],
                    pdf_sha256=doc["pdf_sha256"], policy_id=POLICY,
                    review_method="complete_scope_template_not_human_full_document_review",
                    gross_cash_per_share=float(row.cash_div_tax), bonus_per_share=0.0,
                    historical_feed_available_at=None, formal_training_eligible=False,
                    evidence=evidence)
        decisions.append(dict(normalized_event_id=e["normalized_event_id"], **decision))
    verify()
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("announcement-adjudicate-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out / "reviews.json", dict(policy_id=POLICY, reviews=reviews))
    atomic_json(out / "decisions.json", decisions)
    atomic_json(out / "inputs.json", [dict(path=str(p), sha256=sha) for p, sha in pins])
    report = dict(directory=str(out), at=now(), event_rows=len(events), accepted_scope_reviews=len(reviews),
                  unresolved=len(events)-len(reviews), policy_id=POLICY,
                  refusal_counts=dict(Counter(r for d in decisions for r in d["reasons"])),
                  formal_training_eligible=False, code_sha256=digest(Path(__file__)),
                  artifacts={n:digest(out/n) for n in ["reviews.json", "decisions.json", "inputs.json"]})
    atomic_json(out / "summary.json", report)
    return report
