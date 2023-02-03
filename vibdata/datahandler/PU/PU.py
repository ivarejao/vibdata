from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
import pandas as pd
import numpy as np
from vibdata.datahandler.utils import _get_package_resource_dataframe
import os
from scipy.io import loadmat
from tqdm import tqdm


class PU_raw(RawVibrationDataset, DownloadableDataset):
    """
    Data source: https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/
    LICENSE: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) [https://creativecommons.org/licenses/by-nc/4.0/]
    """
    # mirrors = ["http://groups.uni-paderborn.de/kat/BearingDataCenter"]
    # resources = [('K001.rar', None), ('K005.rar', None), ('KA04.rar', None), ('KA08.rar', None),
    #              ('KA22.rar', None), ('KB27.rar', None), ('KI05.rar', None), ('KI16.rar', None),
    #              ('K002.rar', None), ('K006.rar', None), ('KA05.rar', None), ('KA09.rar', None),
    #              ('KA30.rar', None), ('KI01.rar', None), ('KI07.rar', None), ('KI17.rar', None),
    #              ('K003.rar', None), ('KA01.rar', None), ('KA06.rar', None), ('KA15.rar', None),
    #              ('KB23.rar', None), ('KI03.rar', None), ('KI08.rar', None), ('KI18.rar', None),
    #              ('K004.rar', None), ('KA03.rar', None), ('KA07.rar', None), ('KA16.rar', None),
    #              ('KB24.rar', None), ('KI04.rar', None), ('KI14.rar', None), ('KI21.rar', None)]
    
    #https://drive.google.com/file/d/1PZLt3h1x_rjY6EfWV3yNl3FSDWH4o-Ii/view?usp=sharing
    urls = ["1PZLt3h1x_rjY6EfWV3yNl3FSDWH4o-Ii"]
    resources = [('PU.zip', '1beb53c6fb79436895787e094a22302f')]

    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=PU_raw.resources, download_urls=PU_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=PU_raw.resources)

        self._metainfo = _get_package_resource_dataframe(__package__, "PU.csv")

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        return self._metainfo

    def __getitem__(self, i) -> dict:
        if(isinstance(i, int)):
            data_i = self._metainfo.iloc[i]
            fname, bearing_code = data_i['file_name'], data_i['bearing_code']
            full_fname = os.path.join(self.raw_folder, bearing_code, fname)
            data = loadmat(full_fname, simplify_cells=True)[fname.split('.')[0]]
            sig = PU_raw._getVibration_1(data)
            return {'signal': sig, 'metainfo': data_i}
        elif(isinstance(i, slice)):
            range_idx = list(range(i.start, i.stop, i.step))

            data_i = self._metainfo.iloc[i]
            fname, bearing_code = data_i['file_name'], data_i['bearing_code']
            signal_datas = np.empty(len(range_idx), dtype=object)
            for j in range(len(range_idx)):
                full_fname = os.path.join(self.raw_folder,
                                          bearing_code.iloc[j],
                                          fname.iloc[j])
                data = loadmat(full_fname, simplify_cells=True)[fname.iloc[j].split('.')[0]]
                signal_datas[j] = PU_raw._getVibration_1(data)

            return {'signal': signal_datas, 'metainfo': data_i}
        return super().__getitem__(i)

    @staticmethod
    def _getVibration_1(data):
        Ys = data['Y']
        for y in Ys:
            if(y['Name'] == 'vibration_1'):
                return y['Data']
        raise ValueError

    def asSimpleForm(self):
        metainfo = self.getMetaInfo()
        sigs = []
        files_info = metainfo[['file_name', 'bearing_code']]
        for _, (f, b) in tqdm(files_info.iterrows(), total=len(files_info)):
            full_fname = os.path.join(self.raw_folder, b, f)
            data = loadmat(full_fname, simplify_cells=True)[f.split('.')[0]]
            sigs.append(PU_raw._getVibration_1(data))
        return {'signal': sigs, 'metainfo': metainfo}

    def getLabelsNames(self) -> list:
        return ['Normal', 'Outer ring Fault', 'Inner Ring Fault', 'OR + IR']

    def name(self):
        return "PU"