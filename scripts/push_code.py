"""Upload kneemri/ to Kaggle as your private dataset <user>/kneemri-code (a new version on every run).
Kaggle notebooks attach that dataset, so they run exactly the code in this repo.

    python scripts/push_code.py "what changed"     add --dry-run to only prepare build/kaggle-code
"""
import json, shutil, sys, time
from _common import ROOT, config, git_version, kaggle, wait_for_dataset

dry = '--dry-run' in sys.argv
msg = next((a for a in sys.argv[1:] if not a.startswith('--')), None)
cfg = config()
slug = f"{cfg['kaggle_user']}/{cfg['code_dataset']}"
version = git_version()

stage = ROOT / 'build' / 'kaggle-code'
shutil.rmtree(stage, ignore_errors=True)
shutil.copytree(ROOT / 'kneemri', stage / 'kneemri', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
(stage / 'build_info.json').write_text(json.dumps(dict(commit=version, built=time.strftime('%Y-%m-%d %H:%M:%S')), indent=1), encoding='utf-8')
(stage / 'dataset-metadata.json').write_text(json.dumps(dict(title=cfg['code_dataset'], id=slug, licenses=[dict(name='CC0-1.0')]), indent=1), encoding='utf-8')
print(f'prepared {stage} (code version {version})')
if version.endswith('-dirty'):
    print('note: uncommitted changes - fine while testing; commit once everything works')

if dry:
    print('first upload would run:'); kaggle('datasets', 'create', '-p', stage, '-r', 'zip', dry=True)
    print('later uploads would run:'); kaggle('datasets', 'version', '-p', stage, '-m', msg or f'code {version}', '-r', 'zip', dry=True)
    sys.exit()
status = kaggle('datasets', 'status', slug)
if not status.startswith('FAILED') and any(w in status.lower() for w in ('ready', 'pending')):
    kaggle('datasets', 'version', '-p', stage, '-m', msg or f'code {version}', '-r', 'zip')
else:
    kaggle('datasets', 'create', '-p', stage, '-r', 'zip')
print('\nwaiting for Kaggle to process the upload...')
status = wait_for_dataset(slug)
print(f'\n{slug} is {status}: https://www.kaggle.com/datasets/{slug}')
if status != 'ready': sys.exit('the upload did not become ready; check the dataset page on kaggle.com')
