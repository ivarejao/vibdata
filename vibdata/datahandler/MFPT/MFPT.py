from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from scipy.io import loadmat
from importlib import resources
import os


class MFPT_raw(RawVibrationDataset, DownloadableDataset):
    """
    FIXME: metainfo
    """
    # mirrors = ["https://www.mfpt.org/wp-content/uploads/2020/02/"]
    urls = ["1VxGlOMCEED7jy2qAoE9nKYAK8h5i6TIb"]
    resources = [("MFPT-Fault-Data-Sets-20200227T131140Z-001.zip", '965fa4161fe2c669d375eeb104079d1b')]
    root_dir = 'MFPT Fault Data Sets'

    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=MFPT_raw.resources, download_urls=MFPT_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=MFPT_raw.resources, download_urls=None)

    def __getitem__(self, i) -> dict:
        if(not hasattr(i, '__len__') and not isinstance(i, slice)):
            ret = self.__getitem__([i])
            ret['signal'] = ret['signal'][i]
            ret['metainfo'] = ret['metainfo'].iloc[i]
            return ret
        df = self.getMetaInfo()
        if(isinstance(i, slice)):
            rows = df.iloc[i.start:i.stop:i.step]
        else:
            rows = df.iloc[i]
        file_name = rows['file_name']
        signal_datas = np.empty(len(file_name), dtype=object)
        for i, f in enumerate(file_name):
            data = loadmat(os.path.join(self.raw_folder, MFPT_raw.root_dir, f),
                           simplify_cells=True, variable_names=['bearing'])
            signal_datas[i] = data['bearing']['gs']
        signal_datas = signal_datas

        return {'signal': signal_datas, 'metainfo': rows, 'label_names': self.getLabelsNames()}

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        with resources.path(__package__, "MFPT.csv") as r:
            return pd.read_csv(r)

    def getLabelsNames(self):
        return ['Normal', 'Outer Race', 'Inner Race']
