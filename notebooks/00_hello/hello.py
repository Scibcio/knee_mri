# Phase 0 check: runs on Kaggle, imports the repo code from the kneemri-code dataset and reports what it finds.
import glob, json, os, sys, tarfile, zipfile

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
    raise SystemExit('kneemri-code dataset not attached: add it as an input of this notebook')


code = find_code()
sys.path.insert(0, code)
import kneemri
from kneemri import env

info = glob.glob(f'{INPUT}/*/build_info.json') + glob.glob(f'{INPUT}/*/*/*/build_info.json')
report = dict(code_from=code, kneemri=kneemri.__version__, build=json.load(open(info[0])) if info else None, env=env.describe())
print('code:', report['code_from'], '| kneemri', report['kneemri'], '| build', report['build'])
print('environment:', report['env'])

report['inputs'] = {}
print('\ninputs:')
for r in env.input_roots(INPUT):
    s = env.summarize(r); report['inputs'][os.path.relpath(r, INPUT)] = s
    print(f'  {os.path.relpath(r, INPUT)}: {s["files"]} files, {s["mb"]} MB, {s["types"]}' + (f', DICOM folders {s["dicom_dirs"]}' if s['dicom_dirs'] else ''))

comp = env.competition_dir(INPUT)
report['competition_dir'] = comp
if comp:
    import pandas as pd
    train, ser = pd.read_csv(f'{comp}/train.csv'), pd.read_csv(f'{comp}/train_series.csv')
    report['counts'] = dict(train_studies=len(train), series=len(ser), study_folders=len(os.listdir(f'{comp}/train_series')))
    print('\ncompetition:', comp, report['counts'])
    report['dicom_probe'] = env.probe_dicoms(comp)
    print('\nDICOM compression formats in 300 random series (first file of each):')
    for k, v in report['dicom_probe'].items(): print(f'  {k}: {v}')
else:
    print('\ncompetition data not attached')

json.dump(report, open(f'{WORKING}/hello.json', 'w'), indent=1, default=str)
print('\nsaved hello.json')
