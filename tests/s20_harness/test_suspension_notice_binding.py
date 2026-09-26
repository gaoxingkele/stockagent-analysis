from research.s20_harness.suspension_notice_binding import identity_pages


def test_notice_and_issuer_require_same_page_and_labels():
    assert identity_pages(['证券代码：603003\n公告编号：2025-041'], '603003.SH','2025-041')['document_identity_matched']
    assert not identity_pages(['603003 2025-041'], '603003.SH','2025-041')['document_identity_matched']
    assert not identity_pages(['证券代码：603003','公告编号：2025-041'], '603003.SH','2025-041')['document_identity_matched']
    assert not identity_pages(['证券代码：603003 公告编号：2025-042'], '603003.SH','2025-041')['document_identity_matched']
    assert not identity_pages(['证券代码：603003 公告编号：2025-0419'], '603003.SH','2025-041')['document_identity_matched']
