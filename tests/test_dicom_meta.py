import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from kneemri.dicom_meta import COLUMNS, all_tags, header_row, scan_series


def write_dcm(path, **tags):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = MRImageStorage, generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID, ds.Modality = MRImageStorage, meta.MediaStorageSOPInstanceUID, 'MR'
    ds.Rows, ds.Columns, ds.BitsAllocated, ds.BitsStored, ds.HighBit = 2, 2, 16, 12, 11
    ds.PixelRepresentation, ds.SamplesPerPixel, ds.PhotometricInterpretation = 0, 1, 'MONOCHROME2'
    for k, v in tags.items(): setattr(ds, k, v)
    ds.PixelData = np.zeros((2, 2), np.uint16).tobytes()
    try: ds.save_as(path, enforce_file_format=True)
    except TypeError: ds.save_as(path, write_like_original=False)
    return path


def test_header_row_reads_values(tmp_path):
    p = write_dcm(tmp_path / 'a.dcm', ImagePositionPatient=[1.5, -2, 30], ImageOrientationPatient=[0, 1, 0, 0, 0, -1],
                  PixelSpacing=[0.4, 0.5], InstanceNumber=7, EchoTime=35.5, ImageType=['ORIGINAL', 'PRIMARY'],
                  Manufacturer='ACME')
    row, d = header_row(str(p))
    assert row['error'] is None and d is not None
    assert (row['ipp_x'], row['ipp_y'], row['ipp_z']) == (1.5, -2.0, 30.0)
    assert [row[f'iop_{i}'] for i in range(6)] == [0, 1, 0, 0, 0, -1]
    assert (row['ps_row'], row['ps_col'], row['instance'], row['echo_time']) == (0.4, 0.5, 7, 35.5)
    assert row['image_type'] == 'ORIGINAL\\PRIMARY' and row['transfer_syntax'] == ExplicitVRLittleEndian
    assert row['rows'] == 2 and row['bits_stored'] == 12 and row['photometric'] == 'MONOCHROME2'
    assert set(row) <= set(COLUMNS)


def test_missing_tags_are_none(tmp_path):
    row, _ = header_row(str(write_dcm(tmp_path / 'b.dcm')))
    assert row['ipp_x'] is None and row['iop_5'] is None and row['instance'] is None and row['echo_time'] is None


def test_unreadable_file_is_reported(tmp_path):
    p = tmp_path / 'bad.dcm'
    p.write_bytes(b'not a dicom file at all')
    row, d = header_row(str(p))
    assert d is None and row['error'] and row['size_bytes'] == 23


def test_all_tags_text(tmp_path):
    item = Dataset(); item.CodeValue = 'X'
    p = write_dcm(tmp_path / 'c.dcm', ImageType=['A', 'B'], PatientName='Doe^J', ReferencedImageSequence=Sequence([item, item]))
    t = all_tags(pydicom.dcmread(str(p), stop_before_pixels=True))
    assert t['ImageType'] == 'A\\B' and t['PatientName'] == 'Doe^J'
    assert t['ReferencedImageSequence'] == '<2 items>' and t['TransferSyntaxUID'] == ExplicitVRLittleEndian
    assert 'PixelData' not in t


def test_scan_series(tmp_path):
    folder = tmp_path / 'series1'; folder.mkdir()
    for i in range(3): write_dcm(folder / f'{i}.dcm', InstanceNumber=i + 1, ImagePositionPatient=[0, 0, i * 3.0])
    (folder / 'junk.dcm').write_bytes(b'xx')
    job, rows, tags, counts, n = scan_series(('train', 'study1', 'series1', str(folder)))
    assert job[2] == 'series1' and n == 4 and len(rows) == 4
    assert sum(r['error'] is not None for r in rows) == 1
    assert all(r['study'] == 'study1' and r['split'] == 'train' for r in rows)
    assert counts['InstanceNumber'] == 3 and tags['Modality'] == 'MR'
    assert scan_series(('train', 's', 'missing', str(tmp_path / 'nope')))[4] == 0
