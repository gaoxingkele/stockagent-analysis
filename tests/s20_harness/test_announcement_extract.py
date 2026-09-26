from research.s20_harness.announcement_extract import extract_pages, extract_date_roles, extract_beneficiary_sections
import pytest


def test_cash_header_not_diluted_cash_and_date_mentions_not_role_proof():
    row = dict(ts_code="603081.SH", cash_div_tax=.05, record_date="20240703", ex_date="20240704", pay_date="20240704")
    pages = ["证券代码：603081 A 股每股现金红利0.05元 登记在册的本公司全体股东 2024/7/3 2024/7/4",
             "虚拟分派现金红利0.0493元，回购专用账户不参与。"]
    r = extract_pages(pages, row)
    assert r["cash_status"] == "matches_source"
    assert r["extracted_cash_values"] == ["0.05"]
    assert all(r["expected_date_mentions"].values())
    assert not r["date_role_assignment_verified"] and not r["beneficiary_approved"]
    assert r["cash_evidence"][0]["page"] == 1
    r = extract_pages(pages + ["A股每股现金红利0.06元"], row)
    assert r["cash_status"] == "multiple_values"


@pytest.mark.parametrize("clause", [
    "每10股派现金0.75元人民币（含税，扣税后每10股派现金0.675元）",
    "每10股派0.750000元人民币现金（含税；扣税后每10股派0.675元）",
    "每10股派发现金红利人民币0.75元（含税）",
    "每10股派送现金股利0.75元（含税）",
])
def test_gross_cash_syntax_without_net_cash(clause):
    row = dict(ts_code="000096.SZ", cash_div_tax=.075, record_date="20260811",
               ex_date="20260812", pay_date="20260812")
    result = extract_pages([clause], row)
    assert result["cash_status"] == "matches_source"
    assert len(result["cash_evidence"]) == 1
    assert not result["beneficiary_approved"]


def test_adjusted_header_does_not_silently_override_original_proposal():
    row = dict(ts_code="688651.SH", cash_div_tax=.06027, record_date="20240703",
               ex_date="20240704", pay_date="20240704")
    result = extract_pages([
        "每股分配比例\n每股现金红利0.06027元（含税）",
        "原方案每10股派发现金红利人民币0.6元（含税）。"
        "除息参考每股现金红利0.059元，扣税后每10股派0.54元。",
    ], row)
    assert result["cash_status"] == "multiple_values"
    assert result["extracted_cash_values"] == ["0.06", "0.06027"]
    assert result["cash_evidence"][0]["evidence_kind"] == "distribution_header"
    assert not result["date_role_assignment_verified"]


def test_role_table_and_prose_preserve_conflict_not_date_occurrence():
    row = dict(record_date="20240703", ex_date="20240704", pay_date="20240704")
    table = "股权登记日\n除权（息）日\n现金红利发放日\n2024/7/3\n2024/7/4\n2024/7/4"
    r = extract_date_roles([table], row)
    assert r["all_roles_match_source"] and not r["semantic_approval"]
    r = extract_date_roles([table, "股权登记日为：2024年7月2日"], row)
    assert r["states"]["record_date"] == "conflicting_values"
    assert not r["all_roles_match_source"]
    assert extract_date_roles(["会议2024/7/3申请2024/7/4"], row)["evidence"] == []


def test_payment_clause_and_invalid_or_extra_table_columns():
    row = dict(record_date="20240703", ex_date="20240704", pay_date="20240704")
    prose = "股权登记日为：2024年7月3日；除权除息日：2024年7月4日。现金红利将于2024年7月4日通过股东托管证券公司直接划入其资金账户。"
    assert extract_date_roles([prose], row)["all_roles_match_source"]
    invalid = "股权登记日：2024/2/30"
    assert not extract_date_roles([invalid], row)["evidence"]
    extra = "股权登记日 除权（息）日 新增股份上市日 现金红利发放日 2024/7/3 2024/7/4 2024/7/5 2024/7/4"
    assert not extract_date_roles([extra], row)["all_roles_match_source"]


def test_a_share_table_does_not_shift_listing_into_payment_or_accept_b_share():
    row = dict(record_date="20240703", ex_date="20240704", pay_date="20240708")
    table = "股份类别 股权登记日 最后交易日 除权（息）日 新增无限售条件流通股份上市日 现金红利发放日 Ａ股 2024/7/3 － 2024/7/4 2024/7/5 2024/7/8"
    result = extract_date_roles([table], row)
    assert result["all_roles_match_source"]
    assert result["values"]["pay_date"] == ["20240708"]
    assert not extract_date_roles([table.replace("Ａ股", "B股")], row)["all_roles_match_source"]


def test_four_column_table_and_adjusted_cash_header():
    row = dict(ts_code="688261.SH", cash_div_tax=.17875, record_date="20240703", ex_date="20240704", pay_date="20240708")
    pages = ["每股分配比例，每股转增比例 每股现金红利0.17875元（含税）",
             "股权登记日 除权（息）日 新增无限售条件流通股份上市日 现金红利发放日 2024/7/3 2024/7/4 2024/7/5 2024/7/8"]
    r = extract_pages(pages, row)
    assert r["cash_status"] == "matches_source"
    assert r["date_roles"]["all_roles_match_source"]


def test_beneficiary_section_keeps_cross_page_exclusions_and_refuses_unbounded():
    pages = ["2.分派对象：截至登记日登记在册的本公司全体股",
             "东（公司回购专用证券账户除外）。回购股份不参与分配。3.分配方案：每股0.1元"]
    sections = extract_beneficiary_sections(pages)
    assert len(sections) == 1
    s = sections[0]
    assert s["pages"] == [1, 2] and s["ordinary_holder_phrase_present"]
    assert s["exclusion_or_negation_present"] and s["repurchase_mentioned"]
    assert not s["approved"] and "除外" in s["text"]
    assert not extract_beneficiary_sections([pages[0]])


@pytest.mark.parametrize("start,end", [
    ("三、权益分派对象", "四、权益分派实施方法"),
    ("三、分红派息对象", "四、权益分派方法"),
    ("四、分红派息对象", "五、分配方法"),
    ("四、分红派息对象", "五、分红派息方法"),
    ("四、权益分派对象", "五、权益分派办法"),
    ("（二）分派对象：", "（三）分配方案："),
])
def test_scope_heading_variants_and_exclusion_infix(start, end):
    text = start + "登记在册的除公司回购专用证券账户外的本公司全体股东。" + end
    sections = extract_beneficiary_sections([text])
    assert len(sections) == 1
    assert sections[0]["ordinary_holder_phrase_present"]
    assert sections[0]["exclusion_or_negation_present"]
    assert not sections[0]["approved"]
