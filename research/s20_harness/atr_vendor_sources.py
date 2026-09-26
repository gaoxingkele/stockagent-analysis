"""Join separately receipted Tushare daily/factor parquet snapshots in memory."""
from pathlib import Path
import pandas as pd

from .dependency_receipts import bind,KEYS,BINDINGS
from .label_availability import _instant


def prepare(root,samples,entries,calendar):
    if not isinstance(entries,list) or len(entries)>100000:
        raise ValueError('bounded vendor source bindings required')
    rows=[];seen=set();sample_ids=set(samples.sample_id)
    for e in entries:
        if set(e)!={'sample_id','trading_code','daily','factors'}:
            raise ValueError('exact vendor control binding required')
        if e['sample_id'] in seen or e['sample_id'] not in sample_ids:
            raise ValueError('unique candidate vendor binding required')
        if not isinstance(e['trading_code'],str) or not e['trading_code'].strip():
            raise ValueError('explicit trading code required')
        seen.add(e['sample_id'])
        for name in ['daily','factors']:
            if set(e[name])!=set(BINDINGS):raise ValueError('exact component receipt binding required')
            rows.append(dict(sample_id=e['sample_id'],role='feature',dependency_id=name,**e[name]))
    receipts,evidence=bind(root,pd.DataFrame(rows,columns=KEYS+BINDINGS))
    times=receipts.set_index(['sample_id','dependency_id'])
    meta=samples.set_index('sample_id');packets={};table_cache={}
    for e in entries:
        sid=e['sample_id'];sample=meta.loc[sid]
        clocks={name:_instant(times.loc[(sid,name),'available_at']) for name in ['daily','factors']}
        received=max(clocks.values())
        packet=None
        if received < _instant(sample.prediction_at):
            tables={}
            for name in ['daily','factors']:
                path=(Path(root)/e[name]['artifact_path']).resolve()
                if path.suffix!='.parquet':raise ValueError('vendor snapshots must be parquet')
                required={'ts_code','trade_date','adj_factor'} if name=='factors' else {'ts_code','trade_date','high','low','close','amount'}
                allowed=required if name=='factors' else required|{'open','pre_close','change','pct_chg','vol','ah_vol','ah_amount'}
                key=(path,name)
                if key not in table_cache:
                    table=pd.read_parquet(path)
                    if table.columns.duplicated().any() or not required.issubset(table) or not set(table).issubset(allowed):
                        raise ValueError('explicit Tushare daily/factor fields required')
                    if table[['ts_code','trade_date']].isna().any().any():raise ValueError('missing vendor identities')
                    table_cache[key]=(table.iloc[:0],{code:g for code,g in table.groupby('ts_code',sort=False)})
                empty,groups=table_cache[key]
                table=groups.get(e['trading_code'],empty).copy()
                table['trade_date']=table.trade_date.astype(str)
                if table.trade_date.duplicated().any():raise ValueError('duplicate vendor security/date')
                for date in table.trade_date:
                    close=pd.to_datetime(date,format='%Y%m%d',errors='raise').tz_localize('Asia/Shanghai')+pd.Timedelta(hours=15)
                    if date>sample.signal_date or close>clocks[name]:raise ValueError('future/incomplete vendor row')
                tables[name]=table.loc[table.trade_date.isin(calendar)]
            q=tables['daily'];f=tables['factors']
            merged=q.merge(f[['trade_date','adj_factor']],on='trade_date',how='left',validate='one_to_one').sort_values('trade_date')
            # A missing factor is an unknown window, never implicit factor=1.
            if merged.empty or merged.adj_factor.isna().any():
                packet=dict(schema_version='atr-bars-1',entity_id=sample.entity_id,
                    price_basis='declared_common_adjusted_cny',amount_unit='thousand_cny',bars=[])
            else:
                packet=dict(schema_version='atr-raw-factor-bars-1',entity_id=sample.entity_id,
                    price_basis='unadjusted_cny',amount_unit='thousand_cny',
                    bars=merged[['trade_date','high','low','close','amount','adj_factor']].to_dict('records'))
        packets[sid]=(received,packet)
    return packets,evidence
