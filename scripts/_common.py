import json, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / 'configs' / 'project.json').read_text())
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
    p = subprocess.run(cmd, capture_output=True, text=True)
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
