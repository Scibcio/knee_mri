# Phase 0 check: imports the repo code on Kaggle and reports the environment, the inputs and the DICOM formats.
# INPUT, WORKING and CODE come from notebooks/runner/bootstrap.py, which runs first.
import glob, json, os
import kneemri
from kneemri import env

info = glob.glob(f'{INPUT}/*/build_info.json') + glob.glob(f'{INPUT}/*/*/*/build_info.json')
report = dict(code_from=CODE, kneemri=kneemri.__version__, build=json.load(open(info[0])) if info else None, env=env.describe())
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
