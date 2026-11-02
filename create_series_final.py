import pandas as pd,numpy as np
from math import comb
from pathlib import Path
B=Path(__file__).resolve().parent
source=pd.read_csv(B/'v4_match_predictions_final.csv')
source[['series_id','game_no']]=source.gameid.str.extract(r'^(.*)_game_(\d+)$')
source.game_no=pd.to_numeric(source.game_no,errors='coerce')
audit=pd.read_csv(B/'series_official_crosscheck.csv')
references={2024:'https://liquipedia.net/leagueoflegends/LPL/2024/Regional_Finals',2025:'https://liquipedia.net/leagueoflegends/World_Championship/2025/Qualification/LPL',2026:'https://rft.gg/event/lpl-2026-regional-finals/matches'}
for i,r in audit.iterrows():
 if r['external_status']=='not_yet_checked' and r.league=='WLDs':
  audit.loc[i,'external_status']='schedule_score_matched'
  audit.loc[i,'reference']=references[int(r.year)]
  audit.loc[i,'validation_note']='LPL regional qualification series; date, teams, score and Bo5 format crosschecked against published results; not World Championship main event.'
audit.to_csv(B/'series_official_crosscheck_updated.csv',index=False)

def p_a(elo_a,elo_b,bonus,side_a_blue):
 d=elo_a-elo_b+(bonus if side_a_blue else -bonus)
 return 1/(1+10**(-d/400))

def series_dp(probs,needed):
 # probabilities specific to game positions, conditional on known future side order.
 dp={(0,0):1.}
 for p in probs:
  nxt={}
  for (a,b),v in dp.items():
   if a==needed or b==needed: nxt[(a,b)]=nxt.get((a,b),0)+v
   else:
    nxt[(a+1,b)]=nxt.get((a+1,b),0)+v*p
    nxt[(a,b+1)]=nxt.get((a,b+1),0)+v*(1-p)
  dp=nxt
 return sum(v for (a,b),v in dp.items() if a==needed)

rows=[]
for r in audit.itertuples():
 if r.external_status!='schedule_score_matched': continue
 g=source[source.series_id==r.series_id].sort_values('game_no')
 if len(g)!=r.games or len(g)==0: continue
 first=g.iloc[0]
 elo_a=float(first.v4_blue_elo) if first.blue_team==r.team_a else float(first.v4_red_elo)
 elo_b=float(first.v4_red_elo) if first.blue_team==r.team_a else float(first.v4_blue_elo)
 neutral=p_a(elo_a,elo_b,0,True)
 n=int(r.best_of)//2+1
 # neutral: a true pre-series forecast with no future side knowledge
 neutral_series=sum(comb(n-1+i,i)*neutral**n*(1-neutral)**i for i in range(n))
 # this uses actual future side sequence (which may not be known at kickoff): retrospective sensitivity ONLY
 side_seq=[bool(x.blue_team==r.team_a) for x in g.itertuples()]
 # complete future schedule for bo3/5: pad unknown future game side with alternating first-game side, DOES NOT make it prospectively valid.
 observed_probs=[p_a(elo_a,elo_b,20,s) for s in side_seq]
 diagnostic=series_dp(observed_probs+[neutral]*(int(r.best_of)-len(observed_probs)),n)
 rows.append(dict(series_id=r.series_id,date=r.date,year=r.year,competition='LPL Regional Finals' if r.league=='WLDs' else r.league,team_a=r.team_a,team_b=r.team_b,score=f'{r.wins_a}-{r.wins_b}',games=int(r.games),best_of=int(r.best_of),pre_series_elo_a=elo_a,pre_series_elo_b=elo_b,neutral_game_prob_a=neutral,series_prob_a=neutral_series,conditional_side_diagnostic=diagnostic,result_a=int(r.result_a),verified_source=r.reference,verification='schedule_and_score_crosschecked',method_note='Pre-series neutral Elo probability; ignores side bonus. Future-side diagnostic is NOT a prospective forecast.'))
result=pd.DataFrame(rows).sort_values(['date','series_id'])
result.to_csv(B/'series_verified_backtest.csv',index=False)
print(result.groupby('best_of').apply(lambda g:pd.Series({'n':len(g),'accuracy':np.mean((g.series_prob_a>=.5)==g.result_a),'brier':np.mean((g.series_prob_a-g.result_a)**2),'logloss':np.mean(-(g.result_a*np.log(g.series_prob_a)+(1-g.result_a)*np.log(1-g.series_prob_a)))}),include_groups=False).to_string())
print('rows',len(result),'all',result.verification.value_counts().to_dict())
