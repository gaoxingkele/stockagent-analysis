# Complete announcement scope template policy v1

Research implementation, 2026-09-14. Policy ID: `ordinary_cash_complete_scope_template_v1`.

Current adjudicator emits `ordinary_cash_complete_scope_template_v2`; v1 outputs remain historical evidence. V2 adds only complete issuer-owned repurchase exclusions: infix 除公司回购专用证券账户外的, parenthetical 不含本公司回购专用证券账户 / 本公司回购专用证券账户除外, and the fully matched Shanghai exchange repurchase-guideline sentence specifying that issuer repurchase-account shares do not participate in profit distribution. It does not delete arbitrary exclusion text or infer an issuer identity from an arbitrary company name. All extra clauses, other-holder exclusions and page-number contamination remain unsupported. Existing economic/date/code checks are unchanged. This is an explicitly versioned evidence-policy extension, not an S20 model/selection protocol change.

This is a bounded deterministic alternative to manually approving every ordinary cash event. It is not general semantic understanding, human full-document review, net-tax certification, historical feed-availability proof, or H01 approval. It does not change the frozen model/selection protocol.

## Acceptance contract

The adjudicator pins the extraction summary and normalized source independently, verifies extraction/input hashes, re-extracts all pages from pinned PDF bytes and reconstructs event evidence with the current extractor. Full normalized-row fingerprints must match. A changed parser or evidence bundle requires a new extraction; no stale approvals are reused.

An event must have all of:

- Ordinary SH/SZ share unit, positive gross cash and zero share distribution; CDR and other markets unsupported.

  Code-scope hardening: suffix alone is insufficient. Template acceptance permits only registered current ordinary families SH 600/601/603/605/688 and SZ 000/001/002/003/300/301; B-share 900/200, CDR 689 and unknown families are refused. New families need explicit review, not automatic admission. Existing 386-review union was checked and contains no B-share codes; its one CDR came from separate manual/unit evidence, not this template.
- Usable normalized terms without conflicting variants.
- Matching security code, unique extracted cash consistent with source, and three explicit date-role matches.
- No extracted restructuring, creditor or CDR context.
- Exactly one bounded beneficiary section, fully matching the implemented grammar: named section; cutoff at the record day or exact record date; correct local exchange close and settlement registrar; all registered company shareholders; closing full stop, with no additional qualifications.

This is **whole-section full matching**, not substring recognition of “all shareholders”. Additional sentences, exclusions, unrecognized wording, unknown terms, multiple cash values, share changes and extra date qualifications are refused. Benign repurchase exclusions are deliberately unsupported until a separate complete-clause policy is tested. A reference-price/proposal ambiguity is not resolved merely by choosing the amount matching the vendor.

## Output and integration

The isolated output records policy ID, method, source URL/PDF hash, full-row hash and reconstructed evidence. Accepted scope reviews can be consumed by `distribution_adapter` using `gross_reference_diagnostic`; that adapter still enforces economic terms and conservative timing. `formal_training_eligible` and historical availability remain false. Existing manual reviews are not overwritten or implicitly merged. Unknown events remain in the denominator.

Residual risks: issuer-text extraction errors, unsupported economic qualifications outside a recognized section, source completeness and historical revisions. This narrow policy must be expanded with disconfirming tests and event-level checks, not relaxed until all records pass. Current success on a two-date pilot does not establish coverage of the full event history. Later formal dataset validation must distinguish event-accounting evidence from features available historically.

## Combining evidence without overwriting reviews

`review_merge.build` consumes externally pinned ordered review bundles and the pinned normalized source. It verifies full-row fingerprints and supplied cash, share and date terms against source, then checks overlapping critical scope/unit/tax/timing/group terms for agreement. Any conflict excludes the entire event from the merged accepted bundle and is recorded with all original reviews; neither first nor last writer wins. Compatible reviews use the first source's descriptive metadata, fill only missing compatible critical fields and preserve full originals in the lineage ledger. This does not combine differing source meanings or infer missing tax/unit approval. Inputs are checked again before publishing; source mutation aborts output. The merged bundle is a new isolated artifact and remains ineligible for formal training.
