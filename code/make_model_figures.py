"""Compute the reported model summaries and Figures 6, 7, 9, 10 and 11 from the
retained draws and fitted models. No chain is regenerated here.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, binom
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from types import SimpleNamespace
import baseline_analysis as c
from analysis_core import weibull_b10
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('reproduced'))
args=parser.parse_args()

ROOT=Path(__file__).resolve().parents[1]
OUT=args.output.resolve()/'numerical'; OUT.mkdir(parents=True,exist_ok=True)
FIG=args.output.resolve()/'figures'; FIG.mkdir(parents=True,exist_ok=True)
# Required numeric results are prepared by reproduce_all.py; standalone plotting
# can use retained results without altering them.
for name in ['primary_fit.json','curved_fit.json','lognormal_fit.json','profile_35.json','lifetime_bands.csv']:
    if not (OUT/name).exists():
        import shutil
        shutil.copy2(ROOT/'results'/name,OUT/name)
df,_,_=c.load_input_dataset(ROOT/'data/optocoupler_ttf_unit_level.csv',3.0)
r=SimpleNamespace(df=df,data=c.prepare_data(df),c=c,b10=weibull_b10)
s=np.load(ROOT/'draws/baseline_chains.npy').reshape(-1,3)
cc=np.load(ROOT/'draws/curved_laplace.npy')
ln=np.load(ROOT/'draws/lognormal_laplace.npy')
mle=np.array(json.loads((OUT/'primary_fit.json').read_text())['theta'])
palette=['#245b89','#ad611e','#34776a']
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
 'font.size':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,
 'ytick.labelsize':7,'axes.linewidth':.65,'lines.linewidth':1.2,
 'pdf.fonttype':42,'ps.fonttype':42,'savefig.pad_inches':.025})
def save(fig,name):
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(FIG/f'{name}.png',dpi=360,bbox_inches='tight')
    plt.close(fig)
def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='.9',lw=.45)
def vals(d,p=.1):
    x=np.log(d)
    return [np.exp(s[:,2]-s[:,1]*x+np.log(-np.log1p(-p))/s[:,0]),
            np.exp(cc[:,2]-cc[:,1]*x+cc[:,3]*(x-r.data.xbar)**2+np.log(-np.log1p(-p))/cc[:,0]),
            np.exp(ln[:,2]-ln[:,1]*x+ln[:,0]*norm.ppf(p))]
def survivors(d,N):
    x=np.log(d)
    return [np.exp(-np.exp(np.clip(s[:,0]*(np.log(N)-(s[:,2]-s[:,1]*x)),-745,40))),
            np.exp(-np.exp(np.clip(cc[:,0]*(np.log(N)-(cc[:,2]-cc[:,1]*x+cc[:,3]*(x-r.data.xbar)**2)),-745,40))),
            norm.sf((np.log(N)-(ln[:,2]-ln[:,1]*x))/ln[:,0])]

# Full-data predictive counts are exact posterior mixtures of Binomial(8,p).
frac=[]
for d,g in r.df.groupby('delta_T'):
    horizon=float(g.cycles.max())
    prob=1-survivors(d,horizon)[0]
    pmf=np.mean(binom.pmf(np.arange(9)[:,None],8,prob[None,:]),axis=1)
    cdf=np.cumsum(pmf)
    quant=[int(np.searchsorted(cdf,q)) for q in [.025,.5,.975]]
    frac.append(dict(delta_T=d,horizon=horizon,observed=int(g.event.sum()),
                     lower=quant[0],median=quant[1],upper=quant[2],
                     basis='A' if d<=75 else 'F',pmf=pmf.tolist()))
(OUT/'full_data_counts.json').write_text(json.dumps(frac,indent=2)+'\n')
assert frac[2]['median']==4 and sum(x['observed'] for x in frac)==29

# Low-stress table: B10, median life and both mission durations.
low=[]
for j,model in enumerate(['Weibull log-linear','Curved Weibull','Lognormal']):
    for d in [20,25,35]:
        b=np.quantile(vals(d,.1)[j],[.025,.5,.975]);m=np.quantile(vals(d,.5)[j],[.025,.5,.975])
        low.append(dict(model=model,delta_T=d,b10_lower=b[0],b10_median=b[1],b10_upper=b[2],
                        medianlife_lower=m[0],medianlife_median=m[1],medianlife_upper=m[2],
                        S5000=np.mean(survivors(d,5000)[j]),S10000=np.mean(survivors(d,10000)[j])))
pd.DataFrame(low).to_csv(OUT/'restored_low_stress.csv',index=False)

# Original information criteria, independently recalculated from the same likelihood.
y=r.data.y;x=r.data.x_log;e=r.data.event
le=s[:,2,None]-s[:,1,None]*x[None,:]
ll=e[None,:]*(np.log(s[:,0,None])+(s[:,0,None]-1)*np.log(y)[None,:]-s[:,0,None]*le)-np.exp(np.clip(s[:,0,None]*(np.log(y)[None,:]-le),-745,40))
lppd=np.sum(logsumexp(ll,axis=0)-np.log(len(s)))
pwaic=np.var(ll,axis=0,ddof=1).sum();waic=-2*(lppd-pwaic)
dbar=-2*ll.sum(axis=1).mean();dhat=-2*r.c.loglik_weibull_picm_l(s.mean(axis=0),r.data)
dic=2*dbar-dhat
assert np.isclose(waic,381.2904799585593,atol=2e-6)
assert np.isclose(dic,381.6405369613325,atol=2e-6)
criteria=[]
for name,path,kpar in [('Weibull log-linear','primary_fit.json',3),('Curved Weibull','curved_fit.json',4),('Lognormal','lognormal_fit.json',3)]:
    f=json.loads((OUT/path).read_text());ell=f['loglik']
    criteria.append(dict(model=name,loglik=ell,AIC=2*kpar-2*ell,BIC=kpar*np.log(48)-2*ell,WAIC=waic if kpar==3 and name.startswith('Weibull') else None,DIC=dic if name.startswith('Weibull') else None))
(OUT/'restored_information_criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
del ll,le

# Failure-only descriptive metrics and the high-excursion boundary table.
failed=r.df[r.df.event==1]
fit=np.exp(mle[2]-mle[1]*np.log(failed.delta_T)+np.log(np.log(2))/mle[0])
err=np.log(failed.cycles)-np.log(fit)
rmse=float(np.sqrt(np.mean(err**2)));r2=float(1-np.sum(err**2)/np.sum((np.log(failed.cycles)-np.log(failed.cycles).mean())**2))
assert np.isclose(rmse,.4041044511903653,atol=1e-6)
high=[]
for d in [175,200]:
    b=float(np.exp(mle[2]-mle[1]*np.log(d)+np.log(-np.log(.9))/mle[0]))
    m=float(np.exp(mle[2]-mle[1]*np.log(d)+np.log(np.log(2))/mle[0]))
    high.append(dict(delta_T=d,B10_cycles=b,B10_hours=3*b,median_cycles=m,median_hours=3*m))
pd.DataFrame(high).to_csv(OUT/'restored_high_stress.csv',index=False)
(OUT/'restoration_checks.json').write_text(json.dumps({'WAIC':waic,'DIC':dic,'failure_only_log_RMSE':rmse,'multiplicative_RMSE':np.exp(rmse),'pseudo_R2':r2,'full_data_75_median':frac[2]['median'],'high_stress':high,'all_passed':True},indent=2)+'\n')

# Original stress-life comparison, now showing both kinds of primary uncertainty.
bands=pd.read_csv(OUT/'lifetime_bands.csv');grid=bands.delta_T.to_numpy()
fig,ax=plt.subplots(figsize=(3.48,3.15),constrained_layout=True)
ax.axvspan(20,25,color='.94',zorder=0)
ax.fill_between(grid,bands.new_unit_q025,bands.new_unit_q975,color='.82',alpha=.38)
ax.plot(grid,bands.new_unit_q025,color='.48',ls=':',lw=.75)
ax.plot(grid,bands.new_unit_q975,color='.48',ls=':',lw=.75)
ax.fill_between(grid,bands.median_q025,bands.median_q975,color=palette[0],alpha=.19)
for j,(name,style) in enumerate(zip(['Weibull log-linear','Curved Weibull','Lognormal'],['-','--','-.'])):
    ax.plot(grid,[np.median(vals(d,.5)[j]) for d in grid],color=palette[j],ls=style,label=name)
ax.scatter(failed.delta_T,failed.cycles,s=13,c='.1',edgecolors='white',linewidths=.3,label='Failure',zorder=5)
for i,(d,t,n) in enumerate([(25,1600/3,8),(50,1600/3,8),(75,3200/3,3)]):
    ax.scatter([d],[t],marker='^',s=25,facecolor='white',edgecolor='.12',label='Run-out' if i==0 else None,zorder=6)
    ax.annotate('',(d,t*1.7),(d,t*1.06),arrowprops={'arrowstyle':'->','lw':.7,'color':'.15'})
    ax.text(d+2,t*.79,str(n),fontsize=7,va='top')
ax.set(xlim=(19,153),ylim=(60,200000),xlabel='Thermal excursion ΔT (°C)',ylabel='Lifetime or observed duration (cycles)',xticks=[25,50,75,100,125,150])
ax.set_yscale('log');ax.legend(loc='upper right',frameon=True,framealpha=.95,fontsize=6.7,borderpad=.35,labelspacing=.22)
clean(ax);save(fig,'restored_stress_life')

# Observed-domain failure-fraction plot at the observed horizons.
fig,ax=plt.subplots(figsize=(3.48,2.55),constrained_layout=True)
d=np.array([v['delta_T'] for v in frac]);q=np.array([[v[k] for k in ['lower','median','upper']] for v in frac])/8
ax.errorbar(d+1.6,q[:,1],yerr=[q[:,1]-q[:,0],q[:,2]-q[:,1]],fmt='s',ms=3.6,capsize=2,color=palette[0],label='Predictive median and 95% interval')
ax.scatter(d-1.6,np.array([v['observed'] for v in frac])/8,c='.15',s=18,label='Observed fraction',zorder=6)
ax.set(xlim=(18,157),ylim=(-.04,1.08),xlabel='Thermal excursion ΔT (°C)',ylabel='Fraction failed',xticks=d,yticks=[0,.25,.5,.75,1])
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.29),frameon=False,fontsize=6.9)
clean(ax);save(fig,'restored_failure_fraction')

# Original prior-only versus posterior result, with original-scale ratios in text.
cfg=json.loads((ROOT/'config/publication_analysis.json').read_text())
prior_draws=c.sample_picm_l_prior(cfg['prior_draws'],seed=cfg['prior_seed'])
prior=c.build_prior_vs_data_b10(s,prior_draws,np.array([20.,25.,35.]))[0]
prior.to_csv(OUT/'table_prior_vs_data_b10.csv',index=False)
fig,ax=plt.subplots(figsize=(3.48,2.25),constrained_layout=True)
for i,row in prior.iterrows():
    for off,col,pref,label in [(-.09,palette[0],'post','Posterior'),(.09,palette[1],'prior','Prior only')]:
        m=row[f'{pref}_q50'];lo=row[f'{pref}_q2_5'];hi=row[f'{pref}_q97_5']
        ax.errorbar(m,i+off,xerr=[[m-lo],[hi-m]],fmt='o',ms=3,capsize=2,color=col,label=label if i==0 else None)
ax.set_yticks(range(3),['20 °C','25 °C','35 °C']);ax.invert_yaxis();ax.set_xscale('log')
ax.set_xlabel(r'$B_{10}$ life (cycles; logarithmic scale)');ax.legend(loc='upper left',bbox_to_anchor=(0,1.26),frameon=False,ncol=2)
clean(ax);save(fig,'restored_prior_posterior')

# B10 uncertainty bands for the three models.
fig,ax=plt.subplots(figsize=(3.48,3.15),constrained_layout=True)
ax.axvspan(20,25,color='.94',zorder=0)
for j,(name,style) in enumerate(zip(['Weibull log-linear','Curved Weibull','Lognormal'],['-','--','-.'])):
    v=np.array([np.quantile(vals(d)[j],[.025,.5,.975]) for d in grid])
    pd.DataFrame({'delta_T':grid,'q025':v[:,0],'median':v[:,1],'q975':v[:,2]}).to_csv(OUT/f'figure10_model_{j}.csv',index=False)
    ax.fill_between(grid,v[:,0],v[:,2],color=palette[j],alpha=.13)
    ax.plot(grid,v[:,1],color=palette[j],ls=style,label=name)
for t in [5000,10000]:ax.axhline(t,color='.55',ls=':',lw=.65)
ax.axvline(35,color='.6',ls=':',lw=.7)
ax.set_yscale('log');ax.set(xlim=(20,150),ylim=(60,100000),xlabel='Thermal excursion ΔT (°C)',ylabel=r'$B_{10}$ life (cycles)',xticks=[25,50,75,100,125,150])
ax.legend(frameon=True,fontsize=6.8,framealpha=.95,labelspacing=.25)
clean(ax);save(fig,'restored_model_bands')

# Mission-probability curves; draw-based crossings are tabulated.
fig,ax=plt.subplots(figsize=(3.48,2.75),constrained_layout=True)
dg=np.linspace(10,50,241)
ax.axvspan(10,25,color='.94',zorder=0)
for N,col,sty in [(5000,palette[0],'-'),(10000,palette[1],'--')]:
    curve=[np.mean(r.b10(s,d)>N) for d in dg]
    pd.DataFrame({'delta_T':dg,'probability':curve}).to_csv(OUT/f'figure11_mission_{N}.csv',index=False)
    ax.plot(dg,curve,color=col,ls=sty,label=f'{N:,} cycles')
for prob in [.50,.90,.95]:ax.axhline(prob,color='.6',ls=':',lw=.6)
ax.axvline(25,color='.5',lw=.75)
ax.set(xlim=(10,50),ylim=(0,1.015),xlabel='Thermal excursion ΔT (°C)',ylabel=r'Posterior probability: $B_{10}>N$',xticks=[10,20,25,30,40,50],yticks=[0,.25,.5,.75,1])
ax.legend(frameon=False,loc='lower left');clean(ax);save(fig,'restored_mission_probabilities')

print(json.dumps({'checks':'passed','restored_low_stress_rows':len(low),'full_data_counts':[{k:v[k] for k in ['delta_T','observed','median','lower','upper']} for v in frac],'information_criteria':criteria},indent=2))
