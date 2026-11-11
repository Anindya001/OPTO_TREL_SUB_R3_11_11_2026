"""Regenerate the secondary analyses: comparator draws, group bootstrap, LOSO, Firth-type fit and Bartlett check.

The primary four posterior settings are handled only by run_final_posteriors.py.
This command writes new results separately; it never replaces retained arrays.
Optimisation-based comparator samples may differ across BLAS/platform builds.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.stats import chi2
import baseline_analysis as c

ROOT=Path(__file__).resolve().parents[1]
def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/secondary')
    p.add_argument('--task',choices=['comparators','group-shape','loso','firth-bartlett','all'],default='all')
    a=p.parse_args(); out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((ROOT/'config/publication_analysis.json').read_text())
    df,_,_=c.load_input_dataset(ROOT/'data/optocoupler_ttf_unit_level.csv',3.0);data=c.prepare_data(df)
    fit=c.fit_picm_l_mle(data,cfg['primary_starts'],cfg['optimiser_seed'])
    dump(out/'primary_fit.json',{k:v for k,v in fit.items() if k!='optim_result'})
    if a.task in ['all','comparators']:
        checks=[]
        for name,fitfn,drawfn in [('curved',c.fit_picm_c_sensitivity,c.draw_picm_c_laplace),('lognormal',c.fit_lognormal_aft,c.draw_lognormal_laplace)]:
            f=fitfn(data,cfg[name+'_starts'],cfg[name+'_seed'])
            draws=drawfn(f,cfg['secondary_draws'],cfg[name+'_draw_seed'])['draws']
            np.save(out/(name+'_refitted_laplace.npy'),draws)
            archived=json.loads((ROOT/cfg['comparator_draw_generation_fit_files'][name]).read_text())
            for key in ['theta','cov','cov_pen']:
                if key in archived:archived[key]=np.array(archived[key])
            replay=drawfn(archived,cfg['secondary_draws'],cfg[name+'_draw_seed'])['draws']
            np.save(out/(name+'_laplace.npy'),replay)
            dump(out/(name+'_fit.json'),{k:v for k,v in f.items() if k!='optim_result'})
            old=np.load(ROOT/'draws'/(name+'_laplace.npy'))
            checks.append(dict(model=name,shape=list(draws.shape),exact_array_match=bool(np.array_equal(draws,old)),max_absolute_difference=float(np.max(np.abs(draws-old))),replay_exact_array_match=bool(np.array_equal(replay,old)),replay_max_absolute_difference=float(np.max(np.abs(replay-old))),note='Refitted sample differences and replay from archived full-precision fit are reported separately.'))
        dump(out/'comparator_comparison.json',checks)
        print('Comparator fits and 30,000 draws per model generated.',flush=True)
    if a.task in ['all','group-shape']:
        c.fit_group_specific_weibull_complete(data,float(fit['theta'][0]),n_boot=cfg['group_bootstrap_replicates'],seed=cfg['group_bootstrap_seed']).to_csv(out/'group_shape_bootstrap.csv',index=False)
        print('Group-shape bootstrap complete.',flush=True)
    if a.task in ['all','loso']:
        # LOSO chain and seed settings are those recorded in config/publication_analysis.json.
        loso=c.compute_loso_predictive_adequacy_picm_l(data)
        loso['evaluation_cycles']=[df.loc[df.delta_T==s,'cycles'].max() for s in loso.held_out_stress]
        loso['horizon_basis']=['administrative run-out' if s<=75 else 'largest observed failure time' for s in loso.held_out_stress]
        loso.to_csv(out/'loso.csv',index=False)
        print('LOSO complete.',flush=True)
    if a.task in ['all','firth-bartlett']:
        firth=c.fit_picm_l_firth(data,fit['theta'],cfg['firth_starts'],cfg['firth_seed'])
        dump(out/'firth_fit.json',{k:v for k,v in firth.items() if k!='optim_result'})
        profile=c.profile_likelihood_b10(data,fit['theta'],fit['loglik'],35.,cfg['profile_grid_points'],cfg['profile_seed'])
        bart=c.bartlett_factor_bootstrap(data,fit['theta'],c.logpsi_from_theta(fit['theta'],35.),35.,cfg['bartlett_replicates'],cfg['bartlett_seed'])
        dump(out/'profile_grid.json',profile);dump(out/'bartlett_bootstrap.json',bart)
        lo,hi=c.find_profile_ci(profile['logpsi_grid'],profile['lr_grid']/bart['bartlett_factor'],chi2.ppf(.95,1))
        dump(out/'reference_routes.json',dict(bartlett_interval_cycles=[float(np.exp(lo)),float(np.exp(hi))],firth_B10_cycles=c.b10_use_from_theta(firth['theta'],35.),mle_delta_cycles=c.b10_delta_interval(fit['theta'],fit['cov'],35.)))
        print('Firth and Bartlett diagnostics complete.',flush=True)
    dump(out/'generation_config.json',cfg)
    if a.task=='all':
        from check_secondary_results import check
        if not check(out):raise SystemExit('Secondary result comparison failed; inspect secondary_verification.json')
if __name__=='__main__':main()
