"""Generate manuscript Figures 2 and 8 from the documented programme and results."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('reproduced'))
a=p.parse_args(); ROOT=Path(__file__).resolve().parents[1]
FIG=a.output.resolve()/'figures';FIG.mkdir(parents=True,exist_ok=True)
NUM=a.output.resolve()/'numerical'
if not (NUM/'profile_35.json').exists(): NUM=ROOT/'results'
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':7.5,'axes.labelsize':7.5,'axes.titlesize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'savefig.facecolor':'white'})
colours=['#77509a','#547a24','#a56819','#168080','#b34747','#24577a']
styles=['-',(0,(4,1.5)),(0,(1.2,1.2)),(0,(5,1,1,1)),(0,(3,1,1,1,1,1)),'-']
fig=plt.figure(figsize=(3.45,3.95))
gs=fig.add_gridspec(3,1,height_ratios=[1.05,1,1],left=.17,right=.93,bottom=.15,top=.95,hspace=.75)
ax=fig.add_subplot(gs[0])
for d,c,ls in zip([25,50,75,100,125,150],colours,styles):
 r=d/2.5;ax.plot([0,r,90,90+r,180],[0,d,d,0,0],color=c,ls=ls,lw=1.1)
 ax.text(78,d+3,f'{d}°',ha='center',va='bottom',fontsize=6.5,color=c,bbox={'facecolor':'white','edgecolor':'none','pad':.1})
ax.set(xlim=(0,180),ylim=(-8,177),xticks=[0,60,120,180],yticks=[0,50,100,150],ylabel='Temperature (°C)')
ax.set_title('(a) Six programmed excursions',loc='left',pad=6,fontweight='bold')
ax.text(.98,.97,'0 °C lower set-point\n180 min period',transform=ax.transAxes,ha='right',va='top',fontsize=7.0)
for row,d in [(1,25),(2,150)]:
 ax=fig.add_subplot(gs[row]);r=d/2.5;dw=90-r;c=colours[0 if d==25 else -1]
 ax.plot([0,r,90,90+r,180],[0,d,d,0,0],color=c,lw=1.3)
 ax.set(xlim=(-3,190),ylim=(-.58*d,1.48*d),xticks=[0,60,120,180],yticks=[0,d],ylabel='Temperature (°C)')
 ax.set_title(f'({"b" if d==25 else "c"}) {d} °C excursion',loc='left',pad=6,fontweight='bold')
 for x in [r,90,90+r]:ax.axvline(x,color='.75',lw=.5,ls=':',ymax=.8)
 ax.annotate('',(90,1.14*d),(r,1.14*d),arrowprops={'arrowstyle':'<->','lw':.7,'mutation_scale':5})
 ax.text((r+90)/2,1.23*d,f'Hot dwell {dw:g} min',ha='center',va='bottom',fontsize=7.0)
 ax.annotate('',(180,-.13*d),(90+r,-.13*d),arrowprops={'arrowstyle':'<->','lw':.7,'mutation_scale':5})
 ax.text((90+r+180)/2,-.19*d,f'Cold dwell {dw:g} min',ha='center',va='top',fontsize=7.0)
 if d==25:
  ax.annotate('Heating\n10 min',(5,.5*d),(34,.35*d),fontsize=7.0,ha='center',arrowprops={'arrowstyle':'->','lw':.6})
  ax.annotate('Cooling\n10 min',(95,.55*d),(137,.61*d),fontsize=7.0,ha='center',arrowprops={'arrowstyle':'->','lw':.6})
 else:
  ax.text(43,.2*d,'Heating\n60 min',ha='center',va='center',fontsize=7.0)
  ax.text(149,.71*d,'Cooling\n60 min',ha='center',va='center',fontsize=7.0)
 ax.text(.5,-.47,f'{r:g} + {dw:g} + {r:g} + {dw:g} = 180 min',transform=ax.transAxes,ha='center',fontsize=7.0)
fig.text(.57,.018,'Time within one cycle (min)',ha='center',fontsize=7.5)
for ax in fig.axes:
 ax.spines[['top','right']].set_visible(False);ax.tick_params(length=2.5,width=.6);ax.grid(axis='y',lw=.4,color='.9')
fig.savefig(FIG/'thermal_programme_single.pdf');fig.savefig(FIG/'thermal_programme_single.png',dpi=450);plt.close(fig)
# Five original reference routes, with names preserved and interval status explicit.
ref=pd.read_csv(ROOT/'reference_tables/table_D_use_condition_reliability.csv')
profile=json.loads((NUM/'profile_35.json').read_text())
rows=[]
for i in [0,3,1,2,4]:
 q=ref.iloc[i];est=q.B10_cycles_estimate;lo=q.lower_95;hi=q.upper_95
 if i==3:est=profile['mle'];lo=profile['lower'];hi=profile['upper']
 rows.append((i,est,lo,hi))
labels=['Bayesian','Ordinary profile','Bartlett-rescaled','MLE delta','Firth-type (point)']
fig,ax=plt.subplots(figsize=(3.45,2.45),constrained_layout=True)
for y,(i,m,l,h) in enumerate(rows):
 c='#245b89' if i in [0,3] else '#98662b' if i==1 else '.4'
 if np.isfinite(l):ax.errorbar(m,y,xerr=[[m-l],[h-m]],fmt='o',color=c,capsize=2,ms=3.4)
 else:ax.plot(m,y,'o',color=c,ms=3.4)
ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set(xlim=(0,9000),xticks=[0,3000,6000,9000],xlabel=r'$B_{10}$ at 35 °C (cycles)')
ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',color='.9',lw=.5)
fig.savefig(FIG/'reference_routes.pdf');fig.savefig(FIG/'reference_routes.png',dpi=360);plt.close(fig)
