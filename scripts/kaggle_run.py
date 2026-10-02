"""Run one task on Kaggle, wait for it, and download its output to runs/<task>/.

All tasks run in ONE Kaggle notebook (<user>/kneemri-runner): every run is a new version of it.
The task is notebooks/tasks/<task>.py; notebooks/runner/bootstrap.py is put in front of it.
Settings come from notebooks/runner/kernel-metadata.json; a task can change them with a first line like
    # kaggle: {"enable_internet": "true", "dataset_sources": ["{user}/some-dataset"]}
(lists are added to the defaults, other values replace them).

    python scripts/kaggle_run.py hello        add --dry-run to only prepare build/runner
"""
import json, shutil, sys, time
from _common import ROOT, config, kaggle

dry = '--dry-run' in sys.argv
task = next(a for a in sys.argv[1:] if not a.startswith('--'))
user = config()['kaggle_user']
runner = ROOT / 'notebooks' / 'runner'
code = (ROOT / 'notebooks' / 'tasks' / f'{task}.py').read_text(encoding='utf-8')

meta = json.loads((runner / 'kernel-metadata.json').read_text(encoding='utf-8'))
first = code.splitlines()[0] if code else ''
if first.startswith('# kaggle:'):
    for k, v in json.loads(first[len('# kaggle:'):]).items():
        meta[k] = meta.get(k, []) + v if isinstance(v, list) else v
meta = json.loads(json.dumps(meta).replace('{user}', user))

stage = ROOT / 'build' / 'runner'
shutil.rmtree(stage, ignore_errors=True)
stage.mkdir(parents=True)
bootstrap = (runner / 'bootstrap.py').read_text(encoding='utf-8')
(stage / 'run.py').write_text(f'{bootstrap}\nprint("task: {task}")\n\n{code}', encoding='utf-8')
(stage / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
ref = meta['id']

out = kaggle('kernels', 'push', '-p', stage, dry=dry)
if dry:
    print(f'prepared {stage}'); sys.exit()
if 'error' in out.lower() or out.startswith('FAILED'): sys.exit('push failed, see the message above')

print(f'\nrunning "{task}" on Kaggle: https://www.kaggle.com/code/{ref}')
start, seen_running = time.time(), False
while True:
    time.sleep(30)
    status = kaggle('kernels', 'status', ref).lower()
    seen_running |= any(s in status for s in ('queued', 'running'))
    # right after a push Kaggle can still report the previous run, so a finished status only counts
    # once this run was seen queued/running, or after 3 minutes
    if any(s in status for s in ('complete', 'error', 'cancel')) and (seen_running or time.time() - start > 180): break

dst = ROOT / 'runs' / task
kaggle('kernels', 'output', ref, '-p', dst, '-o')
print(f'\noutput saved in {dst}')
