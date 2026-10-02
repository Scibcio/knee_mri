# Added in front of every task: makes the repo code (the attached kneemri-code dataset) importable on Kaggle.
import glob, os, sys, tarfile, zipfile

INPUT = os.environ.get('KNEEMRI_KAGGLE_INPUT', '/kaggle/input')
WORKING = os.environ.get('KNEEMRI_KAGGLE_WORKING', '/kaggle/working')


def find_code():
    """Folder of the attached kneemri-code dataset that contains kneemri/ (unpacked to /tmp if Kaggle kept it as an archive)."""
    for pat in ['*/kneemri', '*/*/*/kneemri']:
        hits = sorted(glob.glob(f'{INPUT}/{pat}'))
        if hits: return os.path.dirname(hits[0])
    for pat in ['*/kneemri.zip', '*/*/*/kneemri.zip', '*/kneemri.tar', '*/*/*/kneemri.tar']:
        for a in sorted(glob.glob(f'{INPUT}/{pat}')):
            dst = '/tmp/kneemri_code'
            (zipfile.ZipFile(a) if a.endswith('.zip') else tarfile.open(a)).extractall(dst)
            return dst if os.path.isdir(f'{dst}/kneemri') else os.path.dirname(glob.glob(f'{dst}/**/kneemri', recursive=True)[0])
    raise SystemExit('kneemri-code dataset not attached: run scripts/push_code.py first')


CODE = find_code()
sys.path.insert(0, CODE)
