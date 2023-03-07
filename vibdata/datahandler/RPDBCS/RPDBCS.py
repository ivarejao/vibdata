from pathlib import Path
from typing import Dict, Optional
from ..base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from vibdata.definitions import LABELS_PATH


# _labelNameToInt = {
#     'Normal': 0,
#     'Rubbing': 1,
#     'Faulty sensor': 2,
#     'Misalignment': 3,
#     'Unbalance': 4,
# }

def _convert_label_standard(label : str, standard_labels : pd.DataFrame) -> int:
    fixed_name = 'Faulty Sensor' if label == 'Faulty sensor' else label
    return standard_labels.loc[standard_labels['label'] == fixed_name]['id'].values[0]


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
        # where label is column with an int id
        centralized_labels = pd.read_csv(LABELS_PATH)
        dataset_centralized_labels = centralized_labels.loc[centralized_labels['dataset'] == self.name()]
        self._metainfo['label'] = self._metainfo['label'].apply(_convert_label_standard, standard_labels=dataset_centralized_labels)

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
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels : pd.DataFrame = all_labels.loc[all_labels['dataset'] == self.name()]
            dict_labels = {id_label : labels_name for id_label, labels_name, _ in dataset_labels.itertuples()}
            df['label'] = df['label'].apply(lambda id_label : dict_labels[id_label])
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

    def name(self):
        return "RPDBCS"
