"""Reproduce R3 numerical summaries and all 8 computational figure slots without overwriting inputs.

Default: regenerate the four final MCMC settings, verify retained results, refit
core models, and render the publication figures/tables from retained draws.
--from-retained skips MCMC generation. --secondary additionally regenerates
comparator draws, LOSO, group bootstrap, Firth and Bartlett diagnostics.
Experimental setup, recorded traces and X-ray panels are excluded.
"""
from pathlib import Path
import argparse,subprocess,sys,json,shutil,platform,importlib.metadata,time
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'reproduced')
    p.add_argument('--from-retained',action='store_true')
    p.add_argument('--secondary',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    if out==ROOT or any(out==ROOT/d or (ROOT/d) in out.parents for d in ['data','draws','results','figures','config','code','manuscript','reference_tables','historical']):
        p.error('Choose a separate output directory, e.g. reproduced/.')
    if out.exists() and any(out.iterdir()):p.error('Output directory must be empty; use a new name for each run.')
    out.mkdir(parents=True,exist_ok=True);steps=[]
    def run(script,*args):
        cmd=[sys.executable,str(ROOT/'code'/script),*map(str,args)];print('RUN',script,flush=True)
        t=time.monotonic()
        with (out/(script.removesuffix('.py')+'.log')).open('w') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        steps.append(dict(script=script,arguments=list(map(str,args)),returncode=result.returncode,seconds=time.monotonic()-t))
        if result.returncode:raise RuntimeError(f'{script} failed; see {out/script.replace(".py",".log")}')
    report={'python':platform.python_version(),'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','pandas','matplotlib','PyMuPDF']},'mode':('retained summaries' if a.from_retained else 'four posterior regenerations and retained summaries')+('; secondary analyses regenerated' if a.secondary else ''),'steps':steps}
    try:
        run('verify_package.py')
        run('run_final_posteriors.py','--output',out/'posterior',*(['--check-only'] if a.from_retained else ['--save-chains']))
        import verify_results as v
        v.OUTPUT=out/'verification_summary.json'
        if v.main():raise RuntimeError('Statistical verification failed')
        import reproduce_results as r
        r.OUTPUT=out/'numerical';r.main()
        import supplementary_checks as s
        s.RESULTS_DIR=out/'numerical';s.main()
        import r2_checks as r2
        r2.RESULTS_DIR=out;r2.main()
        run('make_model_figures.py','--output',out)
        run('make_protocol_reference_figures.py','--output',out)
        import make_survival_panels as km
        km.OUTPUT=out/'figures';km.main()
        import fitz
        manifest=json.loads((ROOT/'config/figure_manifest.json').read_text())
        for item in manifest:
            dest=out/item['file'];src=out/'figures'/item['publication_basename']
            if item['kind'].startswith('preserved'):shutil.copy2(ROOT/item['file'],dest)
            else:shutil.copy2(src,dest)
            doc=fitz.open(dest);doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(dest.with_suffix('.png'));doc.close()
        run('export_publication_tables.py','--output',out)
        if a.secondary:run('run_secondary_analyses.py','--output',out/'secondary')
        run('verify_package.py','--reproduced',out)
        report['status']='PASS'
    except Exception as exc:
        report['status']='FAIL';report['error']=str(exc);raise
    finally:(out/'execution_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Completed:',out)
if __name__=='__main__':main()
