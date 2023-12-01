from typing import Dict

from vibdata.definitions import LABELS_PATH

from vibdata.raw.base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from vibdata.raw.utils import _get_package_resource_dataframe
import os


class SEU_raw(RawVibrationDataset, DownloadableDataset):
    # https://drive.google.com/file/d/1sEbS-CxL9ZIsIY9_a_ZMU6S7Oavwvhoj/view?usp=sharing
    # https://github.com/cathysiyu/Mechanical-datasets/archive/refs/heads
    mirrors = ["1sEbS-CxL9ZIsIY9_a_ZMU6S7Oavwvhoj"]
    resources = [("SEU.zip", '7800d1f4d6ee404f2d84ff7e60209902')]
    root_dir = os.path.join('gearbox')

    def __init__(self, root_dir: str, download=False):
        if (download):
            super().__init__(root_dir=root_dir, download_resources=SEU_raw.resources, download_urls=SEU_raw.mirrors,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=SEU_raw.resources)

    def _metainfo(self) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "SEU.csv")
        return df

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        metainfo = self._metainfo().copy(False)
        metainfo['file_name'] = metainfo['file_name'].apply(lambda x: [x] * 8)
        metainfo = metainfo.explode('file_name', ignore_index=True)
        metainfo['channel'] = np.arange(len(metainfo)) % 8

        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels['dataset'] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            metainfo['label'] = metainfo['label'].apply(lambda id_label: dict_labels[id_label])
        return metainfo

    def __getitem__(self, i):
        if not hasattr(i, '__len__') and not isinstance(i, slice):
            return self.__getitem__([i])

        if isinstance(i, slice):
            range_idx = list(range(i.start, i.stop, i.step))
            return self.__getitem__(range_idx)

        if isinstance(i, list):
            mi_i = self.getMetaInfo().iloc[i]
            f = mi_i['file_name']
            sig_i = np.empty(len(i), dtype=object)

            for j in range(len(i)):
                full_fname = os.path.join(self.raw_folder, SEU_raw.root_dir, f.iloc[j])
                if 'ball_20_0' in f.iloc[j]:
                    sep = ','
                else:
                    sep = '\t'
                # there is a extra column because of an extra separator
                channels = pd.read_csv(full_fname, sep=sep, skiprows=16, names=['ch' + str(x) for x in range(1, 10)]).values
                channels = channels[:, :8]
                sig_i[j] = channels[:, mi_i.iloc[j]['channel']]
            return {'signal': sig_i, 'metainfo': mi_i}

        mi_i = self.getMetaInfo().iloc[i]
        f = mi_i['file_name'].iloc[0]
        full_fname = os.path.join(self.raw_folder, SEU_raw.root_dir, f)
        if 'ball_20_0' in f:
            sep = ','
        else:
            sep = '\t'
        # there is a extra column because of an extra separator
        channels = pd.read_csv(full_fname, sep=sep, skiprows=16, names=['ch' + str(j) for j in range(1, 10)]).values
        channels = channels[:, :8]
        sig_i = channels[:, mi_i['channel']]
        return {'signal': sig_i, 'metainfo': mi_i}

    def asSimpleForm(self):
        filenames = self._metainfo['file_name']
        sigs = []
        for f in filenames:
            full_fname = os.path.join(self.raw_folder, SEU_raw.root_dir, f)
            if ('ball_20_0' in f):
                sep = ','
            else:
                sep = '\t'
            # there is a extra column because of an extra separator
            channels = pd.read_csv(full_fname, sep=sep, skiprows=16, names=['ch' + str(i) for i in range(1, 10)])
            channels = channels.iloc[:, :8]
            sigs.append(channels.values.T)
        return {'signal': np.vstack(sigs), 'metainfo': self.getMetaInfo()}

    def name(self):
        return "SEU"
