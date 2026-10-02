"""Run a notebook folder from this repo on Kaggle, wait for it, and download its output to runs/<name>/.

    python scripts/kaggle_run.py notebooks/00_hello      add --dry-run to only prepare build/kernels/<name>
"""
import json, shutil, sys, time
from _common import ROOT, config, kaggle

dry = '--dry-run' in sys.argv
folder = ROOT / next(a for a in sys.argv[1:] if not a.startswith('--'))
user = config()['kaggle_user']

meta = json.loads((folder / 'kernel-metadata.json').read_text().replace('{user}', user))
stage = ROOT / 'build' / 'kernels' / folder.name
shutil.rmtree(stage, ignore_errors=True)
shutil.copytree(folder, stage)
(stage / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
ref = meta['id']

out = kaggle('kernels', 'push', '-p', stage, dry=dry)
if dry: sys.exit(f'prepared {stage}')
if 'error' in out.lower() or out.startswith('FAILED'): sys.exit('push failed, see the message above')

print(f'\nrunning on Kaggle: https://www.kaggle.com/code/{ref}')
start, seen_running = time.time(), False
while True:
    time.sleep(30)
    status = kaggle('kernels', 'status', ref).lower()
    seen_running |= any(s in status for s in ('queued', 'running'))
    # right after a push Kaggle can still report the previous run, so a finished status only counts
    # once this run was seen queued/running, or after 3 minutes
    if any(s in status for s in ('complete', 'error', 'cancel')) and (seen_running or time.time() - start > 180): break

dst = ROOT / 'runs' / folder.name
kaggle('kernels', 'output', ref, '-p', dst, '-o')
print(f'\noutput saved in {dst}')
