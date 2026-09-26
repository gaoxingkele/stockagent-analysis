"""Page-addressable extraction diagnostics; text matches are not approval."""
from decimal import Decimal
import json
from pathlib import Path
import re
import unicodedata
import uuid

import pandas as pd
import pymupdf

from .distribution_adapter import row_fingerprint
from .runtime import atomic_json, digest, now


DATE_PATTERN = r"(?<!\d)(20\d{2})\s*(?:年|/|-)\s*(\d{1,2})\s*(?:月|/|-)\s*(\d{1,2})(?!\d)\s*日?"


def extract_date_roles(pages, row):
    """Conservative explicit-role candidates, not historical/semantic approval.

    Tokenize dates before whitespace removal to preserve adjacent table cells.
    Only a fully recognized three-column table or immediate prose assignment
    is supported. Extra columns are not guessed, and all conflicts survive.
    """
    evidence = []
    roles = {"record_date": r"股权登记日", "ex_date": r"(?:除权除息日|除权\(息\)日|除息日)",
             "pay_date": r"现金红利发放日"}
    token = r"<DATE:(\d{8})>"
    for page, original in enumerate(pages, 1):
        normalized = unicodedata.normalize("NFKC", original)
        def replace(match):
            value = f"{int(match[1]):04}{int(match[2]):02}{int(match[3]):02}"
            try:
                pd.to_datetime(value, format="%Y%m%d", errors="raise")
            except ValueError:
                return "<INVALID_DATE>"
            return f"<DATE:{value}>"
        text = re.sub(r"\s+", "", re.sub(DATE_PATTERN, replace, normalized))
        for listing in [False, True]:
            extra_header = r"新增无限售条件流通股份上市日" if listing else ""
            table = (roles["record_date"] + roles["ex_date"] + extra_header
                     + roles["pay_date"] + token * (4 if listing else 3))
            for m in re.finditer(table, text):
                groups = m.groups()
                for role, value in zip(roles, [groups[0], groups[1], groups[-1]]):
                    evidence.append(dict(page=page, role=role, value=value,
                                         kind="four_column_table" if listing else "three_column_table", snippet=m[0]))
        # Explicit A-share table layouts. The last-trading-day placeholder and
        # optional share-listing column are consumed, never shifted into cash.
        for listing in [False, True]:
            extra_header = r"新增无限售条件流通股份上市日" if listing else ""
            pattern = (r"股份类别" + roles["record_date"] + r"最后交易日" + roles["ex_date"]
                       + extra_header + roles["pay_date"] + r"A股" + token + r"[-—–]"
                       + token + (token if listing else "") + token)
            for m in re.finditer(pattern, text):
                groups = m.groups()
                for role, value in zip(roles, [groups[0], groups[1], groups[-1]]):
                    evidence.append(dict(page=page, role=role, value=value,
                                         kind="a_share_table_with_listing" if listing else "a_share_table",
                                         snippet=m[0]))
        for role, label in roles.items():
            # Require an assignment marker so the last table header cannot
            # consume the first column's date as its own prose value.
            for m in re.finditer(label + r"(?:为[:：]?|[:：])" + token, text):
                evidence.append(dict(page=page, role=role, value=m[1], kind="explicit_assignment", snippet=m[0]))
        # Restrict payment inference to the designated cash distribution clause,
        # not any occurrence of a date near a shareholder or an application.
        payment = r"(?:A股股东)?现金红利将于" + token + r"通过.{0,40}直接划入"
        for m in re.finditer(payment, text):
            evidence.append(dict(page=page, role="pay_date", value=m[1], kind="cash_payment_clause", snippet=m[0]))
    values = {role: sorted({e["value"] for e in evidence if e["role"] == role}) for role in roles}
    states = {role: "not_extracted" if not vs else "conflicting_values" if len(vs) > 1 else
              "matches_source" if vs[0] == str(row[role]) else "source_mismatch" for role, vs in values.items()}
    return dict(evidence=evidence, values=values, states=states,
                all_roles_match_source=all(s == "matches_source" for s in states.values()),
                semantic_approval=False)


def extract_beneficiary_sections(pages):
    """Retain whole named scope sections including exclusions across pages.

    Section presence is not approval. In particular a repurchase exclusion is
    not an ordinary-holder exclusion, and unsupported clauses require review.
    """
    parts = [re.sub(r"\s+", "", unicodedata.normalize("NFKC", p)) for p in pages]
    text = "".join(parts)
    offsets, cursor = [], 0
    for part in parts:
        offsets.append((cursor, cursor + len(part)))
        cursor += len(part)
    sections = []
    # SSE numbered subheading and SZSE named major heading. Require a bounded
    # next heading, rather than truncating before a potentially material clause.
    patterns = [r"2[.、]分派对象[:：](.*?)(?=3[.、])",
                r"\(二\)分派对象[:：](.*?)(?=\(三\))",
                r"[一二三四五六七八九十]+[、.](?:权益分派对象|分红派息对象)[:：]?(.*?)(?=[一二三四五六七八九十]+[、.](?:权益分派(?:方法|办法|实施方法|实施办法)|分红派息方法|分配方法))"]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            snippet = m[0]
            involved = [i+1 for i, (start, end) in enumerate(offsets) if start < m.end() and end > m.start()]
            sections.append(dict(pages=involved, text=snippet,
                                 ordinary_holder_phrase_present=bool(re.search(r"登记在册.{0,80}全体股东", snippet)),
                                 exclusion_or_negation_present=bool(re.search(r"除外|除.{0,100}外|不参与|不享有|不包括|不含|仅限", snippet)),
                                 repurchase_mentioned="回购" in snippet,
                                 approved=False))
    return sections


def extract_pages(pages, row):
    cash, dates, beneficiary, risks, codes = [], [], [], [], []
    for number, page in enumerate(pages, 1):
        normalized = unicodedata.normalize("NFKC", page)
        text = re.sub(r"\s+", "", normalized)
        for m in re.finditer(r"证券代码[:：](\d{6})", text):
            codes.append(dict(page=number, value=m[1], snippet=m[0]))
        # Named per-share header avoids confusing diluted reference-price cash,
        # total distributions or tax-specific net cash with holder gross cash.
        amount = r"([0-9]+(?:\.[0-9]+)?)"
        patterns = [
            (r"A股每股现金红利(?:人民币)?" + amount + r"元", Decimal(1), "a_share_header"),
            # Restrict the unqualified per-share value to the named summary
            # heading; do not collect diluted ex-price or after-tax prose.
            (r"每股分配比例(?:,每股转增比例)?[：:]?每股现金红利" + amount + r"元", Decimal(1), "distribution_header"),
            (r"每10股(?:派(?:发|送|息)?|分配)(?:现金(?:红利|股利)?)?(?:人民币)?"
             + amount + r"元(?:人民币)?(?:现金)?\(含税(?=[),;])", Decimal(10), "gross_ten_share_clause"),
        ]
        for pattern, divisor, kind in patterns:
            for m in re.finditer(pattern, text):
                cash.append(dict(page=number, gross_per_share=str(Decimal(m[1]) / divisor),
                                 evidence_kind=kind, snippet=m[0],
                                 context=text[max(0, m.start()-60):m.end()+60]))
        # Preserve cell separators: compacting adjacent table dates can turn
        # '2024/7/3 2024/7/4' into a spurious day 32 and swallow both cells.
        for m in re.finditer(DATE_PATTERN, normalized):
            value = f"{int(m[1]):04}{int(m[2]):02}{int(m[3]):02}"
            try:
                pd.to_datetime(value, format="%Y%m%d", errors="raise")
            except ValueError:
                continue
            dates.append(dict(page=number, value=value, snippet=m[0]))
        for m in re.finditer(r"登记在册.{0,50}全体股东", text):
            beneficiary.append(dict(page=number, snippet=m[0]))
        for token in ["重整", "债权人", "存托凭证", "回购专用", "差异化", "送红股", "转增股本"]:
            if token in text:
                risks.append(dict(page=number, token=token))
    values = sorted({Decimal(c["gross_per_share"]) for c in cash})
    expected = Decimal(str(row["cash_div_tax"]))
    code = str(row["ts_code"]).split(".")[0]
    return dict(code_mentions=codes, code_mentioned=code in {c["value"] for c in codes},
                cash_evidence=cash, extracted_cash_values=[str(v) for v in values],
                cash_status="not_extracted" if not values else "multiple_values" if len(values) > 1 else
                            "matches_source" if values[0] == expected else "source_mismatch",
                date_mentions=dates,
                date_roles=extract_date_roles(pages, row),
                expected_date_mentions={key: str(row[key]) in {d["value"] for d in dates}
                                        for key in ["record_date", "ex_date", "pay_date"]},
                beneficiary_phrase_evidence=beneficiary, review_tokens=risks,
                beneficiary_sections=extract_beneficiary_sections(pages),
                date_role_assignment_verified=False, beneficiary_approved=False,
                scope="phrase/date occurrence only; negation, exclusions, date roles and cash units need semantic review")


def build(root, archive, archive_sha, normalized_path, normalized_sha):
    archive = Path(archive).resolve()
    pins = [(archive / "summary.json", archive_sha), (Path(normalized_path), normalized_sha)]
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("source pin mismatch")
    summary = json.loads((archive / "summary.json").read_text(encoding="utf-8"))
    pins += [(archive / "matches.json", summary["matches_sha256"]), (archive / "receipts.json", summary["receipts_sha256"])]
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("archive metadata changed")
    matches = json.loads((archive / "matches.json").read_text(encoding="utf-8"))
    receipts = json.loads((archive / "receipts.json").read_text(encoding="utf-8"))
    by_url = {r["relative_url"]: r for r in receipts}
    if len(by_url) != len(receipts):
        raise ValueError("duplicate download receipt")
    data = pd.read_parquet(normalized_path).set_index("normalized_event_id", drop=False)
    if not data.index.is_unique:
        raise ValueError("duplicate event rows")
    documents, results = {}, []
    for match in matches:
        row = data.loc[match["normalized_event_id"]]
        if row_fingerprint(row) != match["row_sha256"]:
            raise ValueError("event row changed")
        record = dict(match)
        if match["status"] != "unique_candidate":
            results.append(dict(record, extraction_status="no_unique_document"))
            continue
        url = match["candidate_paths"][0]
        receipt = by_url.get(url)
        if receipt is None or receipt["status"] != "ARCHIVED_NOT_REVIEWED":
            results.append(dict(record, extraction_status="document_not_archived"))
            continue
        p = (archive / receipt["file"]).resolve()
        if not p.is_relative_to(archive) or digest(p) != receipt["sha256"]:
            raise ValueError("PDF source mismatch")
        if url not in documents:
            with pymupdf.open(p) as doc:
                if doc.needs_pass or not 1 <= len(doc) <= 50:
                    raise ValueError("unsupported document")
                pages = [page.get_text() for page in doc]
            documents[url] = dict(pdf_path=str(p), pdf_sha256=receipt["sha256"],
                                  received_at=receipt["received_at"], pages=pages)
            pins.append((p, receipt["sha256"]))
        results.append(dict(record, extraction_status="diagnostic_extracted", evidence=extract_pages(documents[url]["pages"], row)))
    for p, sha in pins:
        if digest(p) != sha:
            raise ValueError("input changed during extraction")
    out = Path(root).resolve() / "output/experiments/s20_safe_v4/sources" / ("announcement-extract-" + uuid.uuid4().hex)
    out.mkdir(parents=True)
    atomic_json(out / "documents.json", documents)
    atomic_json(out / "events.json", results)
    atomic_json(out / "inputs.json", [{"path": str(p.resolve()), "sha256": sha} for p, sha in pins])
    from collections import Counter
    extracted = [r["evidence"] for r in results if "evidence" in r]
    report = dict(directory=str(out), event_rows=len(results), documents=len(documents),
                  cash_status_counts=dict(Counter(e["cash_status"] for e in extracted)),
                  all_three_dates_mentioned=sum(all(e["expected_date_mentions"].values()) for e in extracted),
                  all_date_roles_match_source=sum(e["date_roles"]["all_roles_match_source"] for e in extracted),
                  date_role_status_counts={role: dict(Counter(e["date_roles"]["states"][role] for e in extracted))
                                           for role in ["record_date", "ex_date", "pay_date"]},
                  beneficiary_phrase_present=sum(bool(e["beneficiary_phrase_evidence"]) for e in extracted),
                  beneficiary_section_present=sum(bool(e["beneficiary_sections"]) for e in extracted),
                  formal_event_acceptance=False, at=now(), code_sha256=digest(Path(__file__)),
                  artifacts={name:digest(out/name) for name in ["documents.json", "events.json", "inputs.json"]})
    atomic_json(out / "summary.json", report)
    return report
