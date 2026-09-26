import pytest
from research.s20_harness.suspension_notice_archive import targets, acquire


def test_targets_deduplicate_documents_not_cases():
    case=dict(ts_code='a',trade_date='20240101',source_url='https://static.cninfo.com.cn/finalpage/1.pdf')
    assert len(targets(dict(reviewed_cases=[case,dict(case,trade_date='20240102')])))==1
    with pytest.raises(ValueError,match='duplicate'):
        targets(dict(reviewed_cases=[case,case]))
    with pytest.raises(ValueError,match='URL'):
        targets(dict(reviewed_cases=[dict(case,source_url='https://example.org/1.pdf')]))


@pytest.mark.parametrize('status,body,expected',[(200,b'%PDF-test','ARCHIVED_NOT_REVIEWED'),(200,b'html','FAILED'),(302,b'%PDF-test','FAILED')])
def test_archive_rejects_html_and_redirects(tmp_path,monkeypatch,status,body,expected):
    class Response:
        status_code=status
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def iter_content(self,size): yield body
    def get(url,**kwargs):
        assert kwargs['allow_redirects'] is False
        return Response()
    monkeypatch.setattr('research.s20_harness.suspension_notice_archive.requests.get',get)
    (tmp_path/'pdf').mkdir()
    result=acquire(tmp_path,'https://static.cninfo.com.cn/1.pdf')
    assert result['status']==expected
    assert not result['historical_availability_proven'] and not result['semantic_review_accepted']
