from pathlib import Path
from typing import Dict, Optional
from ..base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np


_labelNameToInt = {
    'Normal': 0,
    'Rubbing': 1,
    'Faulty sensor': 2,
    'Misalignment': 3,
    'Unbalance': 4,
}


class RPDBCS_raw(RawVibrationDataset, DownloadableDataset):
    urls = ["13KHGDnhkF5ZgcpVc90Rajovgwc27alX-"]
    resources = [('RPDBCS.zip', 'b87be8049fc642d57b7de7c631e8e529')]

    def __init__(self, root_dir: str, frequency_domain=False, download=False,
                 **kwargs):
        self.root_dir = root_dir
        self.frequency_domain = True
        self.dataset: Optional[np.ndarray] = None
        self.n_points = 6100

        if download:
            super().__init__(root_dir=root_dir,
                             download_resources=RPDBCS_raw.resources,
                             download_urls=RPDBCS_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir,
                             download_resources=RPDBCS_raw.resources)

        features_dir = Path(self.root_dir).joinpath('RPDBCS_raw', 'RPDBCS',
                                                    'features.csv')
        self._metainfo = pd.read_csv(features_dir, sep=';')

    def _getDataset(self) -> np.ndarray:
        if self.dataset is None:
            dataset_dir = Path(self.root_dir).joinpath('RPDBCS_raw',
                                                       'RPDBCS',
                                                       'spectrum.csv')
            self.dataset = np.loadtxt(dataset_dir, delimiter=';')
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
