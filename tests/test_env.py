import os
from kneemri import env


def make(root, rel, text='x'):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_competition_dir_new_layout(tmp_path):
    f = make(tmp_path, 'competitions/rsna-knee-abnormality-detection/train_series.csv')
    assert env.competition_dir(str(tmp_path)) == str(f.parent)


def test_competition_dir_old_layout(tmp_path):
    f = make(tmp_path, 'rsna-knee-abnormality-detection/train_series.csv')
    assert env.competition_dir(str(tmp_path)) == str(f.parent)


def test_competition_dir_missing(tmp_path):
    assert env.competition_dir(str(tmp_path)) is None


def test_inputs_listed_without_walking_dicoms(tmp_path):
    make(tmp_path, 'competitions/knee/train.csv')
    make(tmp_path, 'competitions/knee/train_series/s1/a/1.dcm')
    make(tmp_path, 'datasets/me/mrnet/train/sagittal/0000.npy')
    make(tmp_path, 'datasets/me/mrnet/train-acl.csv')
    roots = [os.path.relpath(r, tmp_path).replace(os.sep, '/') for r in env.input_roots(str(tmp_path))]
    assert roots == ['competitions/knee', 'datasets/me/mrnet']
    comp = env.summarize(str(tmp_path / 'competitions/knee'))
    assert comp['files'] == 1 and comp['dicom_dirs'] == ['train_series']
    assert env.summarize(str(tmp_path / 'datasets/me/mrnet'))['types'] == {'.npy': 1, '.csv': 1}


def test_describe_runs():
    d = env.describe()
    assert d['python'] and 'numpy' in d
