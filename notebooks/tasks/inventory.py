# Phase 1: read the header of every DICOM file (train + test, no pixels) into tables.
# Options: LIMIT=50 (only the first 50 training studies, for a quick test), WORKERS=16.
# Outputs: slices.parquet (one row per file), series_tags.parquet (all tags of each series' first file),
#          tag_counts.csv (which tags exist, in how many files) and inventory_summary.json.
import json, os, time
from collections import Counter
from multiprocessing import Pool
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from kneemri import env
from kneemri.dicom_meta import COLUMNS, IPP, IOP, PS, FLOATS, INTS, scan_series

LIMIT, WORKERS = PARAMS.get('LIMIT'), PARAMS.get('WORKERS', (os.cpu_count() or 2) * 4)
comp = env.competition_dir(INPUT)

jobs, listed = [], set()
for split in ('train', 'test'):
    ser = pd.read_csv(f'{comp}/{split}_series.csv')
    studies = sorted(ser.StudyInstanceUID.unique())
    if split == 'train' and LIMIT: studies = studies[:LIMIT]
    root = f'{comp}/{split}_series'
    for study in studies:
        on_disk = set(os.listdir(f'{root}/{study}')) if os.path.isdir(f'{root}/{study}') else set()
        in_csv = set(ser.SeriesInstanceUID[ser.StudyInstanceUID == study])
        for s in sorted(in_csv | on_disk):  # also folders that the CSV does not list
            jobs.append((split if s in in_csv else f'{split}-unlisted', study, s, f'{root}/{study}/{s}'))
print(f'{len(jobs)} series to read with {WORKERS} workers' + (f' (LIMIT={LIMIT})' if LIMIT else ''), flush=True)

schema = pa.schema([(c, pa.float64() if c in IPP + IOP + PS + list(FLOATS.values()) else
                     pa.int64() if c in list(INTS.values()) + ['size_bytes'] else pa.string()) for c in COLUMNS])
tag_schema = pa.schema([('split', pa.string()), ('study', pa.string()), ('series', pa.string()),
                        ('keyword', pa.string()), ('value', pa.string())])
files_w = pq.ParquetWriter(f'{WORKING}/slices.parquet', schema, compression='zstd')
tags_w = pq.ParquetWriter(f'{WORKING}/series_tags.parquet', tag_schema, compression='zstd')

rows, tag_rows, counts, series_tag_counts, empty_series = [], [], Counter(), Counter(), []
n_files = n_errors = 0
start = time.time()
with Pool(WORKERS) as pool:
    for done, ((split, study, series, _), r, tags, c, n) in enumerate(pool.imap_unordered(scan_series, jobs, chunksize=4), 1):
        rows += r; counts.update(c); n_files += n; n_errors += sum(x['error'] is not None for x in r)
        if n == 0: empty_series.append(series)
        if tags:
            series_tag_counts.update(tags.keys())
            tag_rows += [dict(split=split, study=study, series=series, keyword=k, value=v) for k, v in tags.items()]
        if len(rows) > 50_000 or done == len(jobs):
            files_w.write_table(pa.Table.from_pylist(rows, schema=schema)); rows = []
            tags_w.write_table(pa.Table.from_pylist(tag_rows, schema=tag_schema)); tag_rows = []
        if done % 500 == 0 or done == len(jobs):
            el = time.time() - start
            print(f'{done}/{len(jobs)} series, {n_files} files, {el / 60:.1f} min, about {el / done * (len(jobs) - done) / 60:.0f} min left', flush=True)
files_w.close(); tags_w.close()
elapsed = time.time() - start

tc = pd.DataFrame({'keyword': list(counts), 'files': list(counts.values())})
tc['series'] = tc.keyword.map(series_tag_counts).fillna(0).astype(int)
tc.sort_values('files', ascending=False).to_csv(f'{WORKING}/tag_counts.csv', index=False)

sl = pd.read_parquet(f'{WORKING}/slices.parquet', columns=['split', 'study', 'series', 'transfer_syntax', 'photometric'])
summary = dict(
    params=dict(LIMIT=LIMIT, WORKERS=WORKERS), minutes=round(elapsed / 60, 1), files_per_second=round(n_files / elapsed, 1),
    studies=sl.groupby('split').study.nunique().to_dict(), series=sl.groupby('split').series.nunique().to_dict(),
    files=int(n_files), unreadable_files=int(n_errors), empty_or_missing_series=len(empty_series),
    empty_or_missing_examples=empty_series[:20], tags_found=len(tc),
    transfer_syntax=sl.transfer_syntax.value_counts(dropna=False).to_dict(),
    photometric=sl.photometric.value_counts(dropna=False).to_dict())
json.dump(summary, open(f'{WORKING}/inventory_summary.json', 'w'), indent=1, default=str)
print('\n' + json.dumps(summary, indent=1, default=str))
print(f'\nsaved slices.parquet, series_tags.parquet, tag_counts.csv, inventory_summary.json in {WORKING}')
