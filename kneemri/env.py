"""Where is the code running, which libraries does it have, and where is the data."""
import glob, importlib, os, platform, sys
from collections import Counter

KAGGLE_INPUT = os.environ.get('KNEEMRI_KAGGLE_INPUT', '/kaggle/input')
DICOM_DIRS = {'train_series', 'test_series'}
LIBS = ['numpy', 'pandas', 'pyarrow', 'sklearn', 'pydicom', 'cv2', 'pylibjpeg', 'libjpeg', 'openjpeg', 'gdcm', 'torch']


def on_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def describe():
    """Python, platform and library versions (None = not installed)."""
    out = dict(python=sys.version.split()[0], platform=platform.platform(), on_kaggle=on_kaggle())
    for name in LIBS:
        try: out[name] = getattr(importlib.import_module(name), '__version__', 'installed')
        except Exception: out[name] = None
    if out['torch']:
        import torch
        out['cuda'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out


def competition_dir(root=KAGGLE_INPUT):
    """Folder holding train_series.csv: <root>/<slug> or <root>/competitions/<slug>."""
    hits = sorted(glob.glob(f'{root}/*/train_series.csv') + glob.glob(f'{root}/*/*/train_series.csv'))
    return os.path.dirname(hits[0]) if hits else None


def input_roots(root=KAGGLE_INPUT):
    """One path per attached input: competitions/<slug>, datasets|notebooks|models/<owner>/<slug>, or old-style <slug>."""
    for kind in sorted(os.listdir(root)):
        p = os.path.join(root, kind)
        if kind == 'competitions':
            yield from (os.path.join(p, x) for x in sorted(os.listdir(p)))
        elif kind in ('datasets', 'notebooks', 'models'):
            for owner in sorted(os.listdir(p)):
                yield from (os.path.join(p, owner, x) for x in sorted(os.listdir(os.path.join(p, owner))))
        else:
            yield p


def summarize(path):
    """File count, size and file types of one input, without walking into the DICOM folders."""
    n, size, types = 0, 0, Counter()
    for d, sub, files in os.walk(path):
        sub[:] = [s for s in sub if s not in DICOM_DIRS]
        for f in files:
            n += 1; size += os.path.getsize(os.path.join(d, f)); types[os.path.splitext(f)[1] or '(none)'] += 1
    return dict(files=n, mb=round(size / 1e6, 1), types=dict(types),
                dicom_dirs=sorted(s for s in DICOM_DIRS if os.path.isdir(os.path.join(path, s))))


def probe_dicoms(comp, n_series=300, seed=0):
    """Open the first file of n random training series: which compression formats occur, and can we decode them?"""
    import pandas as pd, pydicom
    ser = pd.read_csv(f'{comp}/train_series.csv').sample(frac=1, random_state=seed).head(n_series)
    stats = {}
    for study, series in zip(ser.StudyInstanceUID, ser.SeriesInstanceUID):
        d = f'{comp}/train_series/{study}/{series}'
        files = sorted(os.listdir(d)) if os.path.isdir(d) else []
        if not files: stats.setdefault('(series folder missing or empty)', dict(files=0, decoded=0, error=''))['files'] += 1; continue
        ds = pydicom.dcmread(os.path.join(d, files[0]))
        ts = ds.file_meta.get('TransferSyntaxUID')
        s = stats.setdefault(f'{ts.name if ts else "?"} ({ts})', dict(files=0, decoded=0, error=''))
        s['files'] += 1
        try: ds.pixel_array; s['decoded'] += 1
        except Exception as e: s['error'] = s['error'] or f'{type(e).__name__}: {str(e)[:150]}'
    return stats
