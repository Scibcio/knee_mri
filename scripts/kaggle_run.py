"""Run one task on Kaggle, wait for it, and download its output to runs/<task>/.

All tasks run in ONE Kaggle notebook (<user>/kneemri-runner): every run is a new version of it.
The task is notebooks/tasks/<task>.py; notebooks/runner/bootstrap.py is put in front of it.
Settings come from notebooks/runner/kernel-metadata.json; a task can change them with a first line like
    # kaggle: {"enable_internet": "true", "dataset_sources": ["{user}/some-dataset"]}
(lists are added to the defaults, other values replace them).
Task options go after the task name as NAME=VALUE and arrive in the task as PARAMS['NAME'].

    python scripts/kaggle_run.py hello                 add --dry-run to only prepare build/runner
    python scripts/kaggle_run.py inventory LIMIT=50
"""
import ast, json, shutil, sys, time
from _common import ROOT, config, kaggle, wait_for_dataset

dry = '--dry-run' in sys.argv
args = [a for a in sys.argv[1:] if not a.startswith('--')]
task = next(a for a in args if '=' not in a)


def literal(v):
    try: return ast.literal_eval(v)
    except (ValueError, SyntaxError): return v


params = {k: literal(v) for k, v in (a.split('=', 1) for a in args if '=' in a)}
cfg = config()
user = cfg['kaggle_user']
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
(stage / 'run.py').write_text(f'{bootstrap}\nPARAMS.update({params!r})\nprint("task: {task}", PARAMS)\n\n{code}', encoding='utf-8')
(stage / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
ref = meta['id']

for ds in [] if dry else meta['dataset_sources']:  # check the inputs first, so a run never starts without them
    status = wait_for_dataset(ds)
    if status != 'ready':
        hint = 'run `python scripts/push_code.py` first' if ds.endswith('/' + cfg['code_dataset']) else 'check its name, and that you can open it on kaggle.com'
        sys.exit(f'input dataset {ds} is {status}: {hint}')

out = kaggle('kernels', 'push', '-p', stage, dry=dry)
if dry:
    print(f'prepared {stage}'); sys.exit()
if 'error' in out.lower() or out.startswith('FAILED'): sys.exit('push failed, see the message above')
if 'not valid' in out.lower(): sys.exit('Kaggle could not attach an input (message above), so this run will fail: fix that and run again')

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
