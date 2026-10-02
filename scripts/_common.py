import json, pathlib, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / 'configs' / 'project.json').read_text(encoding='utf-8'))
    if cfg['kaggle_user'].startswith('YOUR-'):
        sys.exit('Set your Kaggle username in configs/project.json first.')
    return cfg


def kaggle(*args, dry=False):
    """Run `kaggle <args>`, print its output and return it as text."""
    cmd = ['kaggle', *map(str, args)]
    print('>', ' '.join(cmd))
    if dry: return ''
    if not shutil.which('kaggle'):
        sys.exit('kaggle command not found: activate the .venv and run `pip install -r requirements.txt`.')
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    out = (p.stdout + p.stderr).strip()
    print(out)
    return out if p.returncode == 0 else f'FAILED: {out}'


def git_version():
    """Short commit hash, with '-dirty' if there are uncommitted changes."""
    try:
        h = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return (h or 'no-commit') + ('-dirty' if dirty else '')
    except FileNotFoundError:
        return 'no-git'


def dataset_status(slug):
    """'ready', 'pending', 'missing' or 'error' for one of your Kaggle datasets."""
    out = kaggle('datasets', 'status', slug).lower()
    if out.startswith('failed'): return 'missing' if ('404' in out or 'not found' in out) else 'error'
    return next((s for s in ('ready', 'pending', 'error') if s in out), out)


def wait_for_dataset(slug, minutes=5):
    """Wait while Kaggle is still processing a new dataset version; returns the final status."""
    for _ in range(minutes * 6):
        status = dataset_status(slug)
        if status != 'pending': return status
        time.sleep(10)
    return 'pending'
