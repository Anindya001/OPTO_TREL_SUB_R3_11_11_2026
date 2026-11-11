"""Fail on missing computational assets, invalid hashes or incomplete generated outputs."""
from pathlib import Path
import argparse,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reproduced',type=Path);a=p.parse_args()
    manifest=json.loads((ROOT/'config/figure_manifest.json').read_text());expected=[2,3,6,7,8,9,10,11]
    assert [f['number'] for f in manifest]==expected,'Computational figure index is incomplete'
    required=['README.md','requirements.txt','requirements-figures.txt','config/final_posteriors.json','config/publication_analysis.json','data/optocoupler_ttf_unit_level.csv','code/reproduce_all.py','code/run_secondary_analyses.py','code/make_model_figures.py','code/make_protocol_reference_figures.py','code/make_survival_panels.py','code/export_publication_tables.py']
    for s in ['baseline','diffuse','conservative','relaxed']:required.append('draws/'+s+'_chains.npy')
    required+=['draws/curved_laplace.npy','draws/lognormal_laplace.npy']
    for name in required:assert (ROOT/name).is_file(),f'Missing required file: {name}'
    for item in manifest:
        q=ROOT/item['file'];assert q.is_file(),f'Missing figure: {q}'
        assert hashlib.sha256(q.read_bytes()).hexdigest()==item['sha256'],f'Figure hash differs: {q}'
    sums=ROOT/'SHA256SUMS.txt'
    if sums.exists():
        for line in sums.read_text().splitlines():
            digest,name=line.split(maxsplit=1);q=ROOT/name.lstrip('*')
            assert q.is_file(),f'Missing archived file: {q}'
            assert hashlib.sha256(q.read_bytes()).hexdigest()==digest,f'Checksum mismatch: {q}'
    if a.reproduced:
        out=a.reproduced.resolve()
        for item in manifest:assert (out/item['file']).is_file(),f'Missing reproduced figure: {item["file"]}'
        for i in range(1,13):assert (out/'tables'/f'Table_{i:02d}.csv').is_file(),f'Missing table {i}'
        checks=json.loads((out/'verification_summary.json').read_text());assert checks['failed']==0 and checks['passed']==45
        assert json.loads((out/'numerical/restoration_checks.json').read_text())['all_passed']
    print('PASS: all 8 computational manuscript figures, data, draws and configurations present; hashes valid.')
if __name__=='__main__':main()
