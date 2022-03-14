from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from importlib import resources
import os
from scipy.io import loadmat
from tqdm import tqdm


class XJTU_raw(RawVibrationDataset, DownloadableDataset):
    """
    Data source: https://biaowang.tech/xjtu-sy-bearing-datasets/
    LICENSE: 
    """

    """
    INFOS: "[...]data were saved as a CSV file, in which the first column is the horizontal vibration signals 
    and the second column is the vertical vibration signals."
    DataFrame(columns=['Horizontal_vibration_signals','Vertical_vibration_signals'])
    """
    #https://drive.google.com/file/d/1dWc3YrRiXR5pwUMAhw8lBl5QvQWgv_2R/view?usp=sharing
    urls = ["1dWc3YrRiXR5pwUMAhw8lBl5QvQWgv_2R"]
    resources = [('XJTU-SY_Bearing_Datasets.zip', '8bf2ac5e0c0fc3fb85273e6dbc7da817')]

    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=XJTU_raw.resources, download_urls=XJTU_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=XJTU_raw.resources)

        with resources.path(__package__, "XJTU.csv") as r:
            self._metainfo = pd.read_csv(r)

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        return self._metainfo

    def __getitem__(self, i) -> pd.DataFrame:
        if(not hasattr(i, '__len__') and not isinstance(i, slice)):
            return self.__getitem__([i])
        df = self.getMetaInfo()
        if(isinstance(i, slice)):
            rows = df.iloc[i.start:i.stop:i.step]
        else:
            rows = df.iloc[i]

        file_name = rows['file_name']

        signal_datas = np.empty(len(file_name), dtype=object)
        for i, f in enumerate(file_name):
            full_fname = os.path.join(self.raw_folder, f)
            data = pd.read_csv(full_fname)

            signal_datas[i] = data
        signal_datas = signal_datas

        return {'signal': signal_datas, 'metainfo': rows}

    def asSimpleForm(self):
        metainfo = self.getMetaInfo()
        sigs = []
        files_info = metainfo['file_name']
        for _, (f) in tqdm(files_info.iteritems(), total=len(files_info)):
            full_fname = os.path.join(self.raw_folder, f)
            data = pd.read_csv(full_fname)
            sigs.append(data)
        return {'signal': sigs, 'metainfo': metainfo}

    def getLabelsNames(self):
        return ['Normal', 'Outer Race', 'Cage', 'Inner Race', 'NaN']  