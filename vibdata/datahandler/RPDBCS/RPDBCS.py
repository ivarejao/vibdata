from pathlib import Path
from typing import Dict, Optional
from ..base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from vibdata.definitions import LABELS_PATH


_labelNameToInt = {
    'Normal': 0,
    'Rubbing': 1,
    'Faulty sensor': 2,
    'Misalignment': 3,
    'Unbalance': 4,
}

def _convert_label_standard(label : str, centralized_labels : pd.DataFrame) -> int:
    fixed_name = 'Faulty Sensor' if label == 'Faulty sensor' else label
    return centralized_labels.loc[centralized_labels['label'] == fixed_name]['label'].values[0]


class RPDBCS_raw(RawVibrationDataset, DownloadableDataset):
    urls = ["1ATw19DjzDWbQKOqVKXPOIdOj7v2Vg61V"]
    resources = [('RPDBCS-20221101.zip', '820dd009d911a7929eb769b53ba19b1d')]

    def __init__(self, root_dir: str, frequency_domain=False, download=False,
                 n_points=6100, **kwargs):
        self.root_dir = root_dir
        self.frequency_domain = True
        self.dataset: Optional[np.ndarray] = None
        self.n_points = n_points

        if download:
            super().__init__(root_dir=root_dir,
                             download_resources=RPDBCS_raw.resources,
                             download_urls=RPDBCS_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir,
                             download_resources=RPDBCS_raw.resources)

        features_dir = Path(self.root_dir).joinpath('RPDBCS_raw',
                                                    'RPDBCS-20221101',
                                                    'features.csv')
        self._metainfo = pd.read_csv(features_dir, sep=';')

        # Convert the label column of metainfo to a centralized label standard
        centralized_labels = pd.read_csv(LABELS_PATH)
        self._metainfo['label'] = self._metainfo['label'].apply(_convert_label_standard, std_labels=centralized_labels)

    def _getDataset(self) -> np.ndarray:
        if self.dataset is None:
            dataset_dir = Path(self.root_dir).joinpath('RPDBCS_raw',
                                                       'RPDBCS-20221101',
                                                       'spectrum.npz')
            dataset = []
            with np.load(dataset_dir) as data:
                for signal_id in self._metainfo['id']:
                    signal = data[str(signal_id)][:self.n_points]
                    dataset.append(signal)
            self.dataset = np.array(dataset)
        return self.dataset

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = self._metainfo.copy(False)
        if not labels_as_str:
            df['label'] = df['label'].map(lambda x: _labelNameToInt[x])
        return df

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, i) -> Dict:
        if not hasattr(i, '__len__') and not isinstance(i, slice):
            return self.__getitem__([i])

        dataset = self._getDataset()
        df = self.getMetaInfo().iloc[i]

        if self.frequency_domain:
            sig = dataset[i, :self.n_points]
        else:
            raise NotImplementedError("__getitem__ not implemented for "
                                      "time domain")
        return {'signal': sig, 'metainfo': df}

    def asSimpleForm(self):
        return self[:]

    def getLabelsNames(self):
        return ['Normal', 'Rubbing', 'Faulty sensor', 'Misalignment',
                'Unbalance']

    def name(self):
        return "RPDBCS"
