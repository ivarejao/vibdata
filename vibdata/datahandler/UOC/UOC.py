
from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from importlib import resources
import os
from scipy.io import loadmat
from tqdm import tqdm


class UOC_raw(RawVibrationDataset, DownloadableDataset):
    """
    Data source: https://figshare.com/articles/dataset/Gear_Fault_Data/6127874/1
    LICENSE: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) [https://creativecommons.org/licenses/by-nc/4.0/]
    """      
    urls = ["1oJHir0Faq_kgFnPPMaLSVyBJb6szjEOL"]
    resources = [('UOC_gear_fault_dataset.zip', 'c33f1f6117ee4913257086007790df35')]

    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=UOC_raw.resources, download_urls=UOC_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=UOC_raw.resources)

        with resources.path(__package__, "UOC.csv") as r:
            self._metainfo = pd.read_csv(r)

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        return self._metainfo

    def __getitem__(self, i) -> pd.DataFrame:
        if(not hasattr(i, '__len__') and not isinstance(i, slice)):
            return self.__getitem__([i]).iloc[0]
        df = self.getMetaInfo()
        if(isinstance(i, slice)):
            rows = df.iloc[i.start:i.step:i.stop]
        else:
            rows = df.iloc[i]

        file_name = rows['file_name']
        position = rows['position']
        
        signal_datas = np.empty(len(file_name), dtype=object)
        full_fname = os.path.join(self.raw_folder, file_name[0])
        data = loadmat(full_fname, simplify_cells=True)['AccTimeDomain']

        for i, (f,p) in enumerate(zip(file_name,position)):
            signal_datas[i] = data[:, p]
        signal_datas = signal_datas

        return {'signal': signal_datas, 'metainfo': rows}

    def asSimpleForm(self):
        metainfo = self.getMetaInfo()
        sigs = []
        file_info = ['DataForClassification_TimeDomain.mat','AccTimeDomain']
        full_fname = os.path.join(self.raw_folder, file_info[0])
        sigs = loadmat(full_fname, simplify_cells=True)[file_info[1]]
        return {'signal': sigs, 'metainfo': metainfo}

    def getLabelsNames(self):
        return ['Healthy', 'Missing Tooth', 'Root Crack', 'Spalling', 'Chipping Tip']    