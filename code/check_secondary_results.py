"""Compare regenerated diagnostic results with retained publication values."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def check(out):
    rows=[]
    def add(name,observed,reference,tolerance):
        diff=float(np.max(np.abs(np.asarray(observed)-np.asarray(reference))))
        rows.append(dict(check=name,max_absolute_difference=diff,tolerance=tolerance,passed=diff<=tolerance))
    for name in ['curved','lognormal']:
        add(name+' archived-fit draw replay',np.load(out/(name+'_laplace.npy')),np.load(ROOT/'draws'/(name+'_laplace.npy')),1e-10)
    a=pd.read_csv(out/'group_shape_bootstrap.csv');b=pd.read_csv(ROOT/'reference_tables/group_shape_bootstrap.csv')
    cols=['k_hat','eta_hat_cycles','k_ci_lower','k_ci_upper','bootstrap_bias','k_hat_bias_corrected']
    add('Group bootstrap',a[cols],b[cols],1e-6)
    a=pd.read_csv(out/'loso.csv');b=pd.read_csv(ROOT/'results/loso.csv')
    cols=['held_out_stress','predicted_failure_fraction','predicted_pi_2_5','predicted_pi_97_5']
    add('LOSO reported predictive summaries',a[cols],b[cols],0)
    f=json.loads((out/'firth_fit.json').read_text());ref=pd.read_csv(ROOT/'reference_tables/table_B_picm_l_estimates.csv')
    add('Firth-type estimate',f['theta'],ref.Firth.to_numpy(),1e-6)
    add('Firth-type stationarity (gradient norm)',f['gradient_norm'],0.0,f['gradient_tolerance'])
    add('Firth-type Nelder-Mead confirmation',f['nelder_mead_theta'],f['theta'],1e-6)
    f=json.loads((out/'reference_routes.json').read_text());ref=pd.read_csv(ROOT/'reference_tables/table_D_use_condition_reliability.csv')
    add('Bartlett endpoints (cycles)',f['bartlett_interval_cycles'],ref.iloc[1][['lower_95','upper_95']].to_numpy(dtype=float),.05)
    add('Firth B10 (cycles)',f['firth_B10_cycles'],ref.iloc[4].B10_cycles_estimate,.05)
    add('MLE delta endpoints (cycles)',f['mle_delta_cycles'],ref.iloc[2][['B10_cycles_estimate','lower_95','upper_95']].to_numpy(dtype=float),.05)
    (out/'secondary_verification.json').write_text(json.dumps({'checks':rows,'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows)},indent=2)+'\n')
    for r in rows:print(('PASS' if r['passed'] else 'FAIL'),r['check'],r['max_absolute_difference'])
    return all(r['passed'] for r in rows)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    raise SystemExit(0 if check(a.output.resolve()) else 1)
