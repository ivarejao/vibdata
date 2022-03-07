
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

    def __getitem__(self, i) -> dict:
        if(isinstance(i, int)):
            data_i = self._metainfo.iloc[i]
            fname = data_i['file_name']
            full_fname = os.path.join(self.raw_folder, fname)
            data = loadmat(full_fname, simplify_cells=True)['AccTimeDomain']
            sig = data[:,i]
            return {'signal': sig, 'metainfo': data_i}
        return super().__getitem__(i)

    def asSimpleForm(self):
        metainfo = self.getMetaInfo()
        sigs = []
        file_info = ['DataForClassification_TimeDomain.mat','AccTimeDomain']
        full_fname = os.path.join(self.raw_folder, file_info[0])
        sigs = loadmat(full_fname, simplify_cells=True)[file_info[1]]
        return {'signal': sigs, 'metainfo': metainfo}
