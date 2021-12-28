from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from importlib import resources
import os


class SEU_raw(RawVibrationDataset, DownloadableDataset):
    mirrors = ["https://github.com/cathysiyu/Mechanical-datasets/archive/refs/heads"]
    resources = [("master.zip", '7800d1f4d6ee404f2d84ff7e60209902')]
    root_dir = os.path.join('Mechanical-datasets-master', 'gearbox')

    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=SEU_raw.resources, download_mirrors=SEU_raw.mirrors,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=SEU_raw.resources, download_mirrors=None)

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        with resources.path(__package__, "SEU.csv") as r:
            metainfo = pd.read_csv(r)
        metainfo['file_name'] = metainfo['file_name'].apply(lambda x: [x]*8)
        metainfo = metainfo.explode('file_name', ignore_index=True)
        metainfo['channel'] = np.arange(len(metainfo)) % 8
        return metainfo

    def asSimpleForm(self):
        with resources.path(__package__, "SEU.csv") as r:
            metainfo1 = pd.read_csv(r)
        filenames = metainfo1['file_name']
        sigs = []
        for f in filenames:
            full_fname = os.path.join(self.raw_folder, SEU_raw.root_dir, f)
            if('ball_20_0' in f):
                sep = ','
            else:
                sep = '\t'
            channels = pd.read_csv(full_fname, sep=sep, skiprows=16, names=['ch'+str(i) for i in range(1, 9)])
            sigs.append(channels.values.T)
        return {'signal': np.vstack(sigs), 'metainfo': self.getMetaInfo()}
