# Code made in Pycharm by Igor Varejao
import os
from importlib import resources

import numpy as np
import pandas as pd

from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
from .Tests.Test1 import FirstTest
from .Tests.Test2 import SecondTest
from .Tests.Test3 import ThirdTest


class IMS_raw(RawVibrationDataset, DownloadableDataset):

    source = "https://drive.google.com/file/d/1r9SadjRcUkvI1wJZvi-nu9VzPyQ8oOvE/view?usp=sharing"
    DATAFILE_NAMES = {
        '1stTest': FirstTest.file_names,
        '2ndTest': SecondTest.file_names,
        '3rdTest': ThirdTest.file_names
    }

    urls = mirrors = ["1r9SadjRcUkvI1wJZvi-nu9VzPyQ8oOvE"]  # Google drive id
    resources = [('IMS.zip', '4d24ffef04f5869d68c0bc7cf65ebf77')]

    #
    # Data file organization
    #                  IMS.7z
    #                    |
    #      ----------------------------
    #     |             |             |
    # 1st_test.rar 2nd_test.rar 3rd_test.rar
    #    |             |             |
    #   files        files         files
    #
    # Resources with all the md5sums
    # There are three leves of extraction
    # resources = {'Source': ('IMS.7s', 'd3ca5a418c2ed0887d68bc3f91991f12'),
    #
    #              'Tests': {'1st_test.rar': 'bf1e651c295071a7168fa6fe60c5f214',
    #                        '2nd_test.rar': '32893c492d76c9d3efe9130227f36af5',
    #                        '3rd_test.rar': '11147ea5a16ceaeb5702f3340a72811a'},
    #
    #              'Files': {'1st_test.rar': FirstTest.md5sums_files,
    #                        '2nd_test.rar': SecondTest.md5sums_files,
    #                        '3rd_test.rar': ThirdTest.md5sums_files}
    #              }

    def __init__(self, root_dir: str, download=False):
        if (download):
            super().__init__(root_dir=root_dir, download_resources=IMS_raw.resources, download_urls=IMS_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=IMS_raw.resources, download_mirrors=None)

    # Implement the abstract methods from RawVibrationalDataset
    # ---------------------------------------------------------
    def __getitem__(self, idx) -> dict:
        if (not hasattr(idx, '__len__') and not isinstance(idx, slice)):
            return self.__getitem__([idx]).iloc[0]
        df = self.getMetaInfo()
        if (isinstance(idx, slice)):
            rows = df.iloc[idx.start: idx.step: idx.stop]
        else:
            rows = df.iloc[idx]

        file_name = rows['file_name']
        bear_name = rows['bearing.position']
        signal_datas = np.empty(len(bear_name), dtype=object)

        for i, (f, b) in enumerate(zip(file_name, bear_name)):
            print(f, b)
            print("This is self.raw_folder: ", self.raw_folder)
            file_data = np.genfromtxt(os.path.join(self.raw_folder, f), delimiter=',')
            signal_datas[i] = file_data[:, FirstTest.back_bearing(b)]
        signal_datas = signal_datas

        return {'signal': signal_datas, 'metainfo': rows}

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        with resources.path(__package__, "IMS.csv") as path:
            return pd.read_csv(path)

    def getLabelsNames(self):
        return ['Normal', 'Degraded Outer Race', 'Outer Race', 'Degraded Inner Race', 'Inner Race',
                'Degraded Roller Race', 'Roller Race']
