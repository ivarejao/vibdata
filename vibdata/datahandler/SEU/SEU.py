from typing import Dict
from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from importlib import resources
import os


class SEU_raw(RawVibrationDataset, DownloadableDataset):
    #https://drive.google.com/file/d/1sEbS-CxL9ZIsIY9_a_ZMU6S7Oavwvhoj/view?usp=sharing
    #https://github.com/cathysiyu/Mechanical-datasets/archive/refs/heads
    mirrors = ["1sEbS-CxL9ZIsIY9_a_ZMU6S7Oavwvhoj"]  
    resources = [("Mechanical-datasets-master.zip", '7800d1f4d6ee404f2d84ff7e60209902')]
    root_dir = os.path.join('gearbox')

    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=SEU_raw.resources, download_urls=SEU_raw.mirrors,
                                extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=SEU_raw.resources)
            
    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        with resources.path(__package__, "SEU.csv") as r:
            self._metainfo = pd.read_csv(r)

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        metainfo = self._metainfo.copy(False)
        metainfo['file_name'] = metainfo['file_name'].apply(lambda x: [x]*8)
        metainfo = metainfo.explode('file_name', ignore_index=True)
        metainfo['channel'] = np.arange(len(metainfo)) % 8
        return metainfo

    def __getitem__(self, i):
        if(not isinstance(i, int)):
            return super().__getitem__(i)

        mi_i = self.getMetaInfo().iloc[i]
        f = mi_i['file_name']
        full_fname = os.path.join(self.raw_folder, SEU_raw.root_dir, f)
        if('ball_20_0' in f):
            sep = ','
        else:
            sep = '\t'
        # there is a extra column because of an extra separator
        channels = pd.read_csv(full_fname, sep=sep, skiprows=16, names=['ch'+str(j) for j in range(1, 10)]).values
        channels = channels[:, :8]
        sig_i = channels[:, mi_i['channel']]
        return {'signal': sig_i, 'metainfo': mi_i}

    def asSimpleForm(self):
        filenames = self._metainfo['file_name']
        sigs = []
        for f in filenames:
            full_fname = os.path.join(self.raw_folder, SEU_raw.root_dir, f)
            if('ball_20_0' in f):
                sep = ','
            else:
                sep = '\t'
            # there is a extra column because of an extra separator
            channels = pd.read_csv(full_fname, sep=sep, skiprows=16, names=['ch'+str(i) for i in range(1, 10)])
            channels = channels.iloc[:, :8]
            sigs.append(channels.values.T)
        return {'signal': np.vstack(sigs), 'metainfo': self.getMetaInfo()}
