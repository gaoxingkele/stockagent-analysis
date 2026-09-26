# Ordinary cash dividend integration policy

Research-only implementation note, 2026-09-14. Does not amend frozen S20 model/selection rules or authorize unresolved vendor events.

## Authoritative basis

- [SAT / Caishui 2015 No.101](https://www.chinatax.gov.cn/n810341/n810755/c1797427/content.html): ordinary personal public-market holdings use effective dividend tax rates 20% within one month, 10% over one month through one year, and exemption beyond one year. Shorter-held cash can arrive before final tax deduction on transfer.
- [SAT / Caishui 2012 No.85](https://www.chinatax.gov.cn/chinatax/n810341/n810765/n812151/n812386/c1082457/content.html): remaining operations include FIFO, settlement holdings, and natural calendar month/year holding periods through the day before transfer settlement. The older long-holding tax rate is superseded by the 2015 notice, not reused.
- [Tushare dividend schema](https://tushare.pro/document/2?doc_id=103): cash_div_tax is per-share gross cash, cash_div is described as after-tax; neither establishes a particular investor's final liability. Stock bonus and capitalization rates have separate fields.

These sources were read on the official sites. No individual issuer's rate/beneficiary is certified by the general tax policy. Website retrieval today is not a historical feature availability receipt.

## Accounting contract

Separate gross entitlement, gross payment, accrued tax liability and actual tax debit. Never reduce the cash receipt and debit the tax again. Net economic value subtracts the liability; spendable cash and pending tax remain separate in the trading ledger.

`cash_tax.tax_bounds` handles explicitly scoped unrestricted SH/SZ personal single-lot cash dividends after the policy effective boundary. Inputs require actual acquisition and transfer-settlement dates, record date and gross entitlement. It does not infer settlement from trade date or turn 20 trading days into a fixed month. Unknown exit returns possible tax rates; missing calendar anniversaries at month-end/leap boundaries retain uncertainty until operational settlement rules are supplied. FIFO multi-lot, restricted shares, institutions, BSE and bonus taxation require separate handling.

The function is not yet connected to economic-window liabilities or formal labels. For cash-only category integration, source-term validation, no conflicting/multiple unresolved entitlement, share-rate zero, issuer identity, cash unit, beneficiary and complete event scope must be established. Generic policy must not stamp fabricated per-event beneficiary reviews. First compare gross/net-liability bounds on source-bound events; preserve class ambiguity where the bounds cross profitability or risk thresholds. Additional missing events cannot be bounded merely by knowing the tax rate.
