"""Pinned reference event replay, not a formal H02 or market-fill executor."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import uuid

from . import portfolio_book as book_api
from . import portfolio_cash_actions as cash_api
from . import portfolio_share_actions as share_api
from .corporate_actions import Distribution
from .exit_policy import ExitPolicy
from .portfolio_valuation import PriceMark, ShareCreditMark, value_portfolio
from .runtime import atomic_json, digest, now, load_plan, canonical
from .portfolio_evidence import bind as bind_evidence


def step(book, packet):
    if set(packet) != {"kind", "args"} or not isinstance(packet["args"], dict):
        raise ValueError("event requires explicit kind/args")
    args = dict(packet["args"])
    kind = packet["kind"]
    calls = {"buy_request": book_api.reserve_buy, "buy_result": book_api.settle_buy,
             "exit_result": book_api.settle_portfolio_exit,
             "cash_accrual": cash_api.accrue_cash_claim, "cash_receipt": cash_api.receive_cash_claim,
             "dividend_tax_settlement": cash_api.settle_dividend_tax,
             "share_receipt": share_api.receive_share_claim,
             "share_basis_allocation": share_api.allocate_share_basis,
             "share_position_transfer": share_api.transfer_share_position,
             "fractional_share_settlement": share_api.settle_fractional_share}
    if kind in calls:
        return calls[kind](book, **args)
    if kind == "exit_request":
        args["policy"] = ExitPolicy(**args["policy"])
        return book_api.reserve_exit(book, **args)
    if kind == "cash_record":
        args["distribution"] = Distribution(**args["distribution"])
        return cash_api.record_cash_claim(book, **args)
    if kind == "share_record":
        args["distribution"] = Distribution(**args["distribution"])
        return share_api.record_share_claim(book, **args)
    if kind == 'mixed_distribution_record':
        args['distribution'] = Distribution(**args['distribution'])
        return share_api.record_mixed_distribution(book, **args)
    if kind == 'follow_on_eligibility':
        from .share_eligibility import register
        args['target_distribution']=Distribution(**args['target_distribution'])
        return register(book,**args)
    if kind == "valuation":
        event_id = args.pop("event_id")
        at = book_api.clock(book, event_id, args["knowledge_at"])
        args["marks"] = [PriceMark(**m) for m in args["marks"]]
        args['share_marks'] = [ShareCreditMark(**m) for m in args.get('share_marks',[])]
        report = value_portfolio(book, **args)
        return book_api.append(book, event_id, at, dict(kind="valuation", report=report))
    raise ValueError("unsupported replay event: " + str(kind))


def build(root, input_path, input_sha):
    input_path = Path(input_path).resolve()
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha or "") or digest(input_path) != input_sha:
        raise ValueError("replay input pin mismatch")
    plan = json.loads(input_path.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "1" or plan.get("evidence_mode") not in {"synthetic", "supplied_reference"}:
        raise ValueError("explicit replay schema/evidence mode required")
    if not isinstance(plan.get("events"), list) or not 1 <= len(plan["events"]) <= 10000:
        raise ValueError("bounded nonempty event stream required")
    book = book_api.create(plan["initial_cash"], plan["calendar"])
    sources = {input_path: input_sha}
    evidence_pins,evidence_report=bind_evidence(root,plan)
    if input_path in evidence_pins: raise ValueError('event input cannot be its own evidence')
    sources.update(evidence_pins)
    sources.update({p: digest(p) for p in Path(__file__).parent.glob("*.py")})
    def check():
        if any(digest(p) != h for p, h in sources.items()):
            raise ValueError("replay source changed")
    check()
    out = Path(root).resolve()/"output/experiments/s20_safe_v4/sources"/("portfolio-replay-"+uuid.uuid4().hex)
    out.mkdir(parents=True)
    blobs = out/"code_blobs"; blobs.mkdir()
    for p, h in sources.items():
        if p.suffix == ".py":
            (blobs/h).write_bytes(p.read_bytes())
            if digest(blobs/h) != h:
                raise ValueError("replay code snapshot changed")
    atomic_json(out/"input.json", plan)
    atomic_json(out/"inputs.json", {str(p): h for p, h in sources.items()})
    snapshots = [book_api.accounting_snapshot(book)]
    completed = 0
    def checkpoint(status, error=None):
        atomic_json(out/"checkpoint.json", dict(status=status, completed_events=completed,
            next_event_index=completed, book=asdict(book), error=error,
            resume_supported=False, formal_training_authorized=False))
    checkpoint("RUNNING")
    try:
        for packet in plan["events"]:
            check()
            proposed = step(book, packet)
            snapshot = book_api.accounting_snapshot(proposed)
            check()
            book = proposed
            snapshots.append(snapshot)
            completed += 1
            checkpoint("RUNNING")
        atomic_json(out/"journal.json", list(book.journal))
        atomic_json(out/"accounting_snapshots.json", snapshots)
        checkpoint("COMPLETED_DIAGNOSTIC")
    except Exception as exc:
        checkpoint("FAILED", dict(type=type(exc).__name__, message=str(exc)))
        raise
    report = dict(directory=str(out), at=now(), evidence_mode=plan["evidence_mode"],
        **evidence_report,
        completed_events=completed, journal_rows=len(book.journal), final_accounting=snapshots[-1],
        terminal_buy_orders=len(book.buys), terminal_exit_orders=len(book.exits),
        terminal_unpaid_cash_claims=sum(not c.paid and c.gross_amount > 0 for c in book.cash_claims),
        terminal_held_units=sum(p.remaining_quantity for p in book.positions),
        terminal_unallocated_share_credits=sum(r.quantity for c in book.share_claims if c.transferred_position_id is None for r in c.receipts),
        unknown_valuation_events=sum(e["kind"] == "valuation" and e["report"]["nav"] is None for e in book.journal),
        event_stream_completion_is_not_position_resolution=True,
        independent_execution_evidence_verified=False, formal_H02_passed=False,
        formal_training_authorized=False, status="COMPLETED_DIAGNOSTIC",
        artifacts={n:digest(out/n) for n in ("input.json", "inputs.json", "checkpoint.json",
                                            "journal.json", "accounting_snapshots.json")})
    atomic_json(out/"summary.json", report)
    return report


def verify(root, directory, summary_sha):
    """Re-execute the pinned event stream, without writing or claiming fills."""
    root,directory=Path(root).resolve(),Path(directory).resolve()
    if not directory.is_relative_to(root): raise ValueError('portfolio replay escapes root')
    if digest(directory/'summary.json')!=summary_sha: raise ValueError('portfolio summary pin mismatch')
    summary=load_plan(directory/'summary.json')
    expected={'input.json','inputs.json','checkpoint.json','journal.json','accounting_snapshots.json'}
    if set(summary['artifacts'])!=expected: raise ValueError('portfolio artifact set mismatch')
    def artifacts():
        if digest(directory/'summary.json')!=summary_sha or any(digest(directory/n)!=h for n,h in summary['artifacts'].items()):
            raise ValueError('portfolio artifacts changed')
    artifacts()
    pins=load_plan(directory/'inputs.json')
    codes={str(p) for p in Path(__file__).parent.glob('*.py')}
    if {p for p in pins if Path(p).suffix=='.py'}!=codes:
        raise ValueError('portfolio code scope mismatch')
    plan=load_plan(directory/'input.json')
    evidence_pins,evidence_report=bind_evidence(root,plan)
    if any(pins.get(str(p))!=h for p,h in evidence_pins.items()): raise ValueError('portfolio evidence pins missing')
    originals=[p for p in pins if Path(p).suffix!='.py' and Path(p) not in evidence_pins]
    if len(originals)!=1: raise ValueError('one original event input required')
    def sources():
        for p,h in pins.items():
            if digest(Path(p))!=h: raise ValueError('portfolio source changed')
            if Path(p).suffix=='.py' and digest(directory/'code_blobs'/h)!=h:
                raise ValueError('portfolio code blob changed')
    sources()
    plan=load_plan(directory/'input.json')
    if canonical(plan)!=canonical(load_plan(originals[0])):
        raise ValueError('portfolio original input differs')
    if (plan.get('schema_version')!='1' or plan.get('evidence_mode') not in {'synthetic','supplied_reference'}
        or not isinstance(plan.get('events'),list) or not 1<=len(plan['events'])<=10000):
        raise ValueError('invalid bounded portfolio event plan')
    book=book_api.create(plan['initial_cash'],plan['calendar'])
    snapshots=[book_api.accounting_snapshot(book)]
    for packet in plan['events']:
        book=step(book,packet)
        snapshots.append(book_api.accounting_snapshot(book))
    def equal(name,value):
        saved=json.loads((directory/name).read_text(encoding='utf-8'))
        if canonical(saved)!=canonical(value): raise ValueError('portfolio reconstruction mismatch: '+name)
    equal('journal.json',list(book.journal))
    equal('accounting_snapshots.json',snapshots)
    equal('checkpoint.json',dict(status='COMPLETED_DIAGNOSTIC',completed_events=len(plan['events']),
        next_event_index=len(plan['events']),book=asdict(book),error=None,resume_supported=False,
        formal_training_authorized=False))
    checks=dict(evidence_mode=plan['evidence_mode'],**evidence_report,completed_events=len(plan['events']),journal_rows=len(book.journal),
        final_accounting=snapshots[-1],terminal_buy_orders=len(book.buys),terminal_exit_orders=len(book.exits),
        terminal_unpaid_cash_claims=sum(not c.paid and c.gross_amount>0 for c in book.cash_claims),
        terminal_held_units=sum(p.remaining_quantity for p in book.positions),
        terminal_unallocated_share_credits=sum(r.quantity for c in book.share_claims if c.transferred_position_id is None for r in c.receipts),
        unknown_valuation_events=sum(e['kind']=='valuation' and e['report']['nav'] is None for e in book.journal),
        event_stream_completion_is_not_position_resolution=True,independent_execution_evidence_verified=False,
        formal_H02_passed=False,formal_training_authorized=False,status='COMPLETED_DIAGNOSTIC')
    if any(canonical(summary.get(k))!=canonical(v) for k,v in checks.items()):
        raise ValueError('portfolio summary reconstruction mismatch')
    sources();artifacts()
    return dict(events_recomputed=len(plan['events']),journal_and_checkpoints_verified=True,
                independent_execution_evidence_verified=False,formal_H02_passed=False,formal_training_authorized=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--input-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[2], args.input, args.input_sha256), indent=2))
