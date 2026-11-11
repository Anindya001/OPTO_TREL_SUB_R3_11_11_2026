"""Export the 12 manuscript table sources, with explicit numerical provenance."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
from analysis_core import load_dataset
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reproduced');a=p.parse_args()
    out=a.output.resolve();n=out/'numerical';t=out/'tables';t.mkdir(parents=True,exist_ok=True);index=[]
    def save(number,title,df,basis):
        name=f'Table_{number:02d}.csv';df.to_csv(t/name,index=False);index.append(dict(number=number,title=title,file=name,basis=basis))
    save(1,'Thermal programme',pd.DataFrame(json.loads((n/'thermal_programme.json').read_text())['profiles']),'Calculated from protocol')
    data=load_dataset();rows=[]
    for d,g in data.groupby('delta_T_C'):
        rows.append(dict(delta_T_C=d,devices=len(g),failures=int(g.event.sum()),censored=int((1-g.event).sum()),failure_hours='; '.join(f'{x:g}' for x in g.loc[g.event==1,'ttf_hours']),censor_hours='; '.join(f'{x:g}' for x in g.loc[g.event==0,'ttf_hours']),regime='S' if d<=50 else 'T' if d==75 else 'F'))
    save(2,'Observed outcomes',pd.DataFrame(rows),'Calculated from the 48 measured lifetime records')
    par=pd.read_csv(n/'parameters.csv');mle=json.loads((n/'primary_fit.json').read_text())['theta'];ref=pd.read_csv(ROOT/'reference_tables/table_B_picm_l_estimates.csv')
    par['MLE']=mle;par['Firth']=ref.Firth.to_numpy()
    save(3,'Parameter estimates',par,'MLE and posterior recalculated; Firth-type estimates from the secondary run; regeneration: --secondary')
    save(4,'LOSO predictive adequacy',pd.read_csv(ROOT/'results/loso.csv'),'Retained train-only MCMC predictive summaries; regeneration: --secondary')
    save(5,'Information criteria',pd.DataFrame(json.loads((n/'restored_information_criteria.json').read_text())),'Recomputed likelihood and posterior criteria')
    save(6,'Separate group shapes',pd.read_csv(ROOT/'reference_tables/group_shape_bootstrap.csv'),'Retained 2000-replicate group bootstrap; regeneration: --secondary')
    save(7,'Common-shape likelihood-ratio check',pd.DataFrame([json.loads((n/'common_shape.json').read_text())]),'Recomputed likelihood-ratio check')
    ref=pd.read_csv(ROOT/'reference_tables/table_D_use_condition_reliability.csv');profile=json.loads((n/'profile_35.json').read_text());ref.loc[3,['B10_cycles_estimate','lower_95','upper_95']]=[profile['mle'],profile['lower'],profile['upper']]
    b35=pd.read_csv(out/'posterior/prior_sensitivity.csv').query("setting=='baseline' and delta_T==35.0").iloc[0]  # same final draws as Tables 9 and 10
    ref.loc[0,['B10_cycles_estimate','lower_95','upper_95']]=[b35['median'],b35['q025'],b35['q975']]
    ref=ref[['method','delta_T_use','B10_cycles_estimate','lower_95','upper_95']].copy()
    ref['interval_status']=['Posterior credible interval','Supporting rescaling check','Asymptotic delta confidence interval','Asymptotic profile confidence interval','Point-estimate check only']
    save(8,'Reference-case estimation routes',ref.iloc[[0,3,1,2,4]],'Bayesian row from the final baseline draws and direct ordinary profile recomputed; Firth-type point from the secondary run; delta and Bartlett reference values retained; regeneration: --secondary')
    save(9,'Low-stress summaries',pd.read_csv(n/'restored_low_stress.csv'),'Recomputed from retained primary and comparator draws')
    prior=pd.read_csv(out/'posterior/prior_sensitivity.csv')
    save(10,'Final four-prior sensitivity',prior,'Recomputed from the four final retained posterior arrays')
    save(11,'Mission-threshold crossings',pd.read_csv(n/'crossings.csv'),'Recomputed from retained primary posterior')
    save(12,'High-stress boundary checks',pd.read_csv(n/'restored_high_stress.csv'),'Recomputed from MLE')
    (t/'table_manifest.json').write_text(json.dumps(index,indent=2)+'\n')
    print('Exported all 12 manuscript table sources.')
if __name__=='__main__':main()
