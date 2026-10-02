"""Read DICOM headers (never the pixels) into flat rows. Used by the Phase 1 inventory."""
import os
from collections import Counter
import pydicom
from pydicom.multival import MultiValue

# Values that can change from slice to slice are kept for every file; everything else is kept once per series.
IPP, IOP, PS = ['ipp_x', 'ipp_y', 'ipp_z'], [f'iop_{i}' for i in range(6)], ['ps_row', 'ps_col']
FLOATS = {'SliceLocation': 'slice_location', 'SliceThickness': 'slice_thickness', 'SpacingBetweenSlices': 'slice_spacing',
          'EchoTime': 'echo_time', 'TriggerTime': 'trigger_time', 'RescaleSlope': 'rescale_slope',
          'RescaleIntercept': 'rescale_intercept', 'WindowCenter': 'window_center', 'WindowWidth': 'window_width'}
INTS = {'InstanceNumber': 'instance', 'Rows': 'rows', 'Columns': 'cols', 'NumberOfFrames': 'frames',
        'EchoNumbers': 'echo_number', 'AcquisitionNumber': 'acquisition', 'BitsStored': 'bits_stored',
        'PixelRepresentation': 'pixel_rep'}
TEXTS = {'ImageType': 'image_type', 'PhotometricInterpretation': 'photometric', 'AcquisitionTime': 'acq_time',
         'ContentTime': 'content_time'}
COLUMNS = (['split', 'study', 'series', 'file', 'size_bytes', 'transfer_syntax'] + IPP + IOP + PS
           + list(FLOATS.values()) + list(INTS.values()) + list(TEXTS.values()) + ['error'])


def _first(v):
    if v is None or isinstance(v, (str, bytes)): return v
    try: return v[0] if len(v) else None
    except TypeError: return v


def _float(v):
    try: return float(_first(v))
    except (TypeError, ValueError): return None


def _int(v):
    f = _float(v)
    return int(f) if f is not None and f == f and abs(f) < 2**62 else None


def _text(v):
    if v is None: return None
    if isinstance(v, bytes): return f'<{len(v)} bytes>'
    if isinstance(v, (list, tuple, MultiValue)): return '\\'.join(map(str, v))
    return str(v)


def _floats(v, n):
    try:
        vals = [float(x) for x in v]
        return vals if len(vals) == n else [None] * n
    except (TypeError, ValueError):
        return [None] * n


def header_row(path):
    """(row of per-file values, dataset or None). Unreadable files get a row with `error` set."""
    row = {'file': os.path.basename(path), 'size_bytes': os.path.getsize(path), 'error': None}
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:
        row['error'] = f'{type(e).__name__}: {str(e)[:200]}'
        return row, None
    ts = getattr(d, 'file_meta', {}).get('TransferSyntaxUID')
    row['transfer_syntax'] = str(ts) if ts else None
    for key, cols in (('ImagePositionPatient', IPP), ('ImageOrientationPatient', IOP), ('PixelSpacing', PS)):
        row.update(zip(cols, _floats(d.get(key), len(cols))))
    for key, col in FLOATS.items(): row[col] = _float(d.get(key))
    for key, col in INTS.items(): row[col] = _int(d.get(key))
    for key, col in TEXTS.items(): row[col] = _text(d.get(key))
    return row, d


def all_tags(d):
    """keyword -> short text for every top-level element; sequences shown as '<n items>'."""
    out = {}
    for el in d:
        key = el.keyword or str(el.tag)
        out[key] = f'<{len(el.value)} items>' if el.VR == 'SQ' else (_text(el.value) or '')[:200]
    ts = getattr(d, 'file_meta', {}).get('TransferSyntaxUID')
    if ts: out['TransferSyntaxUID'] = str(ts)
    return out


def scan_series(job):
    """job = (split, study, series, folder) -> (job, rows, tags of its first readable file, keyword counts, number of files)."""
    split, study, series, folder = job
    rows, tags, counts = [], None, Counter()
    files = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    for f in files:
        row, d = header_row(os.path.join(folder, f))
        row.update(split=split, study=study, series=series)
        rows.append(row)
        if d is not None:
            counts.update({el.keyword or str(el.tag) for el in d})
            if tags is None: tags = all_tags(d)
    return job, rows, tags, counts, len(files)
