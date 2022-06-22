# Code made in Pycharm by Igor Varejao
import os
from vibdata.datahandler.utils import _get_package_resource_dataframe

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

    def __init__(self, root_dir: str, download=False, with_thirdtest=False):
        if (download):
            super().__init__(root_dir=root_dir, download_resources=IMS_raw.resources, download_urls=IMS_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=IMS_raw.resources, download_mirrors=None)
        self.third_test = with_thirdtest

    def __getTest(self, idx):
        if (idx < 17250):
            return '1st_test'
        elif (17250 <= idx < 21184):
            return '2nd_test'
        elif self.third_test:
            return '3rd_test'
        else:
            raise IndexError(f"{idx} out of range")

    # Implement the abstract methods from RawVibrationalDataset
    # ---------------------------------------------------------
    def __getitem__(self, idx) -> dict:
        try:
            if (not hasattr(idx, '__len__') and not isinstance(idx, slice)):
                # return self.__getitem__([idx]).iloc[0]
                return self.__getitem__([idx])
            df = self.getMetaInfo()
            if (isinstance(idx, slice)):
                rows = df.iloc[idx.start: idx.stop: idx.step]
                range_idx = list(range(idx.start, idx.stop, idx.step))
            else:
                rows = df.iloc[idx]
                range_idx = idx
        except IndexError as error:
            print(f"{error} with idx={idx}")
            exit(1)

        file_name = rows['file_name']
        bear_name = rows['bearing.position']
        signal_datas = np.empty(len(bear_name), dtype=object)

        for i, (f, b, fidx) in enumerate(zip(file_name, bear_name, range_idx)):
            file_data = np.loadtxt(os.path.join(self.raw_folder, f"{self.__getTest(fidx)}/{f}"), delimiter='\t', unpack=True)
            signal_datas[i] = file_data[FirstTest.back_bearing(b), :]
        signal_datas = signal_datas

        return {'signal': signal_datas, 'metainfo': rows}

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "IMS.csv")
        if self.third_test:
            return df
        else:
            return df[:21184]

    def getLabelsNames(self):
        return ['Normal', 'Degraded Outer Race', 'Outer Race', 'Degraded Inner Race', 'Inner Race',
                'Degraded Roller Race', 'Roller Race']
