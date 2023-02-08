
from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from vibdata.datahandler.utils import _get_package_resource_dataframe
import os
from scipy.io import loadmat
from tqdm import tqdm


class EAS_raw(RawVibrationDataset, DownloadableDataset):
    """
    Data source: https://fordatis.fraunhofer.de/handle/fordatis/151.2
    LICENSE: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) [https://creativecommons.org/licenses/by-nc/4.0/]
    """ 
    urls = ["1YK8isJkibkCdjKwoc00POsxllFxQdh6_"]
    resources = [('EAS.zip', 'dff193b9ee04e2203b565bf2e635cb77')]

    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=EAS_raw.resources, download_urls=EAS_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=EAS_raw.resources)

        self._metainfo = _get_package_resource_dataframe(__package__, "EAS.csv")

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
        first_position = rows['first_position']
        last_position = rows['last_position']

        signal_datas = np.empty(len(file_name), dtype=object)

        uniques_files = file_name.unique()
        for i, uf in enumerate(uniques_files):
            full_fname = os.path.join(self.raw_folder, uf)
            data = pd.read_csv(full_fname)[['Vibration_1','Vibration_2','Vibration_3']]
            for i, (f,fp,lp) in enumerate(zip(file_name, first_position, last_position)):
                if f == uf:
                    signal_datas[i] = data[fp:lp+1]
        signal_datas = np.hstack(signal_datas).T

        return {'signal': signal_datas, 'metainfo': rows}

    def asSimpleForm(self):
        metainfo = self.getMetaInfo()
        sigs = []
        files_info = metainfo['file_name']
        for _, (f) in tqdm(files_info.iteritems(), total=len(files_info)):
            full_fname = os.path.join(self.raw_folder, f)
            data = pd.read_csv(full_fname)
            sigs.append(data[['Vibration_1','Vibration_2','Vibration_3']])
        return {'signal': sigs, 'metainfo': metainfo}

    def name(self):
        return "EAS"