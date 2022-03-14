from typing import Iterator
import pytest
import numpy as np
import pandas as pd
from tempfile import TemporaryDirectory, tempdir


def test_download():
    from vibdata.datahandler import MFPT_raw
    with TemporaryDirectory() as temp_dir:
        D = MFPT_raw(temp_dir, download=True)
        sig = D[0]['signal']

        assert(len(sig) == 585936)
        assert(round(np.max(sig)-3.708878, 5) == 0)
        assert(len(D.getMetaInfo()) == 20)


###### DATA ########
@pytest.fixture
def example_data_signals() -> np.ndarray:
    sigs = np.empty((3, 512), dtype=float)
    for i in range(3):  # three samples
        x = np.linspace(0.0, 1.0, 512, endpoint=False)
        y = i*np.sin(30.0 * 2.0*np.pi*x) + 0.5*np.sin(40.0 * 2.0*np.pi*x)
        sigs[i] = y

    return sigs


@pytest.fixture
def example_metainfo() -> pd.DataFrame:
    return pd.DataFrame({'field1': [1, 1, 2], 'label': [0, 1, 0]})


@pytest.fixture
def example_data1(example_data_signals, example_metainfo) -> dict:
    return {'signal': example_data_signals, 'metainfo': example_metainfo}


@pytest.fixture
def example_data1_iterator(example_data1):
    class _ExampleDataset:
        def __getitem__(self, i):
            sigs = example_data1['signal'][i]
            metainfo = example_data1['metainfo'].iloc[i]
            return {'signal': sigs, 'metainfo': metainfo}

        def __len__(self):
            return len(example_data1)

        def __iter__(self):
            for i in range(len(self)):
                yield self[i]

    return _ExampleDataset()
#####################


def testTransform1(example_data1: dict):
    from vibdata.datahandler.transforms.signal import FFT, SelectFields, Sequential

    transforms = Sequential([FFT(), SelectFields(['signal'], ['label'])])
    ret = transforms.transform(example_data1)
    assert((ret['label'] == example_data1['metainfo']['label']).all())
    sig = ret['signal']
    assert(round(np.std(sig)-0.086328714, 5) == 0)


def testTransform2(example_data1_iterator: Iterator):
    from vibdata.datahandler.transforms.TransformDataset import transform_and_saveDataset, PickledDataset
    from vibdata.datahandler.transforms.signal import SelectFields

    transforms = SelectFields(['signal'], ['label'])

    with TemporaryDirectory() as temp_dir:
        D1 = transform_and_saveDataset(example_data1_iterator, transforms=transforms,
                                       dir_path=temp_dir, batch_size=1024)
        D2 = transform_and_saveDataset(example_data1_iterator, transforms=transforms,
                                       dir_path=temp_dir, batch_size=1024)
        assert(len(D2) == len(example_data1_iterator))
        assert((D1[0]['label'] == D2[0]['label']).all())
