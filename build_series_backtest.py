"""Conservative V4 series audit. No inferred series is called independently verified.
Requires v4_match_predictions_final.csv; preserves source data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from math import comb

BASE = Path(__file__).resolve().parent
source = pd.read_csv(BASE/'v4_match_predictions_final.csv')
source['date'] = pd.to_datetime(source['date'], errors='coerce')
ids = source['gameid'].astype(str).str.extract(r'^(?P<series_id>.+)_game_(?P<game_no>\d+)$')
source = pd.concat([source,ids],axis=1)
source['game_no']=pd.to_numeric(source['game_no'],errors='coerce')
rows=[]
for sid,g in source.dropna(subset=['series_id']).groupby('series_id'):
    g=g.sort_values('game_no'); reasons=[]
    teams=set(g['blue_team']).union(g['red_team'])
    if len(teams)!=2: reasons.append('not exactly two teams')
    if g['game_no'].duplicated().any() or list(g['game_no']) != list(range(1,len(g)+1)): reasons.append('nonconsecutive game numbers')
    if g['league'].nunique()!=1: reasons.append('mixed competitions')
    if g['date'].isna().any() or (g.date.max()-g.date.min()).total_seconds()>60*60*72: reasons.append('date gap >72h / missing')
    if len(teams)==2 and not all({x.blue_team,x.red_team}==teams for x in g.itertuples()): reasons.append('team mismatch')
    if len(teams)==2:
        first_team=g.iloc[0]['blue_team']
        wins=sum((r.blue_team==first_team and int(r.blue_win)==1) or (r.red_team==first_team and int(r.blue_win)==0) for r in g.itertuples())
    else: first_team=None;wins=0
    losses=len(g)-wins
    maxwins=max(wins,losses)
    if maxwins not in (2,3): reasons.append('no completed Bo3/Bo5 winner')
    if maxwins in (2,3):
        needed=maxwins
        # series must terminate at the game where winning team reaches target
        winner=first_team if wins>losses else next(iter(teams-{first_team})) if len(teams)==2 else None
        win_seq=[(r.blue_team==winner and int(r.blue_win)==1) or (r.red_team==winner and int(r.blue_win)==0) for r in g.itertuples()] if winner else []
        if win_seq and sum(win_seq[:-1])>=needed: reasons.append('winner clinched before final recorded game')
        if len(g)>2*needed-1: reasons.append('too many games for series format')
    if len(g)<2: reasons.append('single game cannot establish a series')
    first=g.iloc[0]
    p=float(first.v4_adjusted_prob)
    if not (0<=p<=1): reasons.append('invalid initial probability')
    best_of=2*maxwins-1 if maxwins in (2,3) else None
    series_p = sum(comb(needed-1+i,i)*p**needed*(1-p)**i for i in range(needed)) if best_of and not reasons else np.nan
    # Probability for first-game Blue team using first-game pre-game Elo and +20 assumed constant for subsequent games
    # Caveat: actual sides can change, which makes constant first-blue probability imperfect.
    rows.append(dict(series_id=sid,year=int(first.year),league=first.league,date=str(first.date.date()),team_a=first_team,team_b=(list(teams-{first_team})[0] if len(teams)==2 else ''),games=len(g),wins_a=wins,wins_b=losses,best_of=best_of or '',pre_series_p_a=p,series_win_prob_a=series_p,result_a=int(wins>losses) if wins!=losses else np.nan,status='candidate' if not reasons else 'excluded',reasons='; '.join(reasons)))
report=pd.DataFrame(rows).sort_values(['year','date','series_id'])
report.to_csv(BASE/'series_audit.csv',index=False)
valid=report[report.status=='candidate'].copy()
valid.to_csv(BASE/'series_candidates.csv',index=False)
print('total match rows',len(pd.read_csv(BASE/'v4_match_predictions_final.csv')))
print('series with patterned ids',len(report),'candidate',len(valid),'excluded',len(report)-len(valid))
print('candidate by year/format:\n',valid.groupby(['year','best_of']).size().to_string())
print('Caution: candidate formats inferred from score, not independently verified from schedule. Do not claim validated series performance.')
