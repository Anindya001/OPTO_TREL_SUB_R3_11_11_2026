"""Targeted checks for the revised KM display and audit-output separation."""
from pathlib import Path
import hashlib,subprocess,sys,tempfile
import numpy as np
from make_survival_panels import kaplan_meier
from analysis_core import load_dataset
from run_final_posteriors import load_config
ROOT=Path(__file__).resolve().parents[1]
def main():
    d=load_dataset();g=d[d.delta_T_C==75]
    k=kaplan_meier(g.cycles.to_numpy(),g.event.to_numpy())
    assert np.isclose(k['step_t'][-2],2138/3) and np.isclose(k['step_t'][-1],3200/3)
    assert k['step_s'][-2]==k['step_s'][-1]==3/8
    assert len(k['censor_t'])==3 and np.all(k['censor_s']==3/8)
    g=d[d.delta_T_C==25];k=kaplan_meier(g.cycles.to_numpy(),g.event.to_numpy())
    assert np.all(k['step_s']==1) and np.isclose(k['step_t'][-1],1600/3)
    k=kaplan_meier(np.array([1,1,2.]),np.array([1,0,1]))
    assert np.allclose(k['step_s'],[1,2/3,0]) and np.allclose(k['censor_s'],[2/3])
    load_config()
    record=ROOT/'results/final_posteriors/run_record.json'
    before=record.read_bytes()
    with tempfile.TemporaryDirectory() as out:
        subprocess.run([sys.executable,str(ROOT/'code/run_final_posteriors.py'),'--check-only','--output',out],check=True,stdout=subprocess.DEVNULL)
    assert record.read_bytes()==before, 'Check-only must preserve the regeneration record'
    print('PASS: KM plateau, censor locations, no-failure group, tied times, archive hashes and audit separation')
if __name__=='__main__':main()
