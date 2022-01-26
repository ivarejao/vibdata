# Code made in Pycharm by Igor Varejao
from importlib import resources

import pandas as pd

from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset
from Tests.Test1 import FirstTest
from Tests.Test2 import SecondTest
from Tests.Test3 import ThirdTest

class IMS_raw(RawVibrationDataset, DownloadableDataset):

    source = "https://ti.arc.nasa.gov/c/3/"
    # Data file organization
    #           IMS.7z
    #             |
    #   --------------------
    #   |         |        |
    # 1stTest/ 2ndTest/ 3rdTest/
    #   ...      ...      ...
    #
    DATAFILE_NAMES = {
        '1stTest': FirstTest.file_names,
        '2ndTest': SecondTest.file_names,
        '3rdTest': ThirdTest.file_names
    }

    mirrors = ["https://ti.arc.nasa.gov/c/3/"]
    resources = [('IMS.7s', 'd3ca5a418c2ed0887d68bc3f91991f12')]

    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=IMS_raw.resources, download_mirrors=IMS_raw.mirrors)
        else:
            super().__init__(root_dir=root_dir, download_resources=IMS_raw.resources, download_mirrors=None)

    # Implement the abstract methods from RawVibrationalDataset
    # ---------------------------------------------------------
    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        with resources.path(__package__, "IMS.csv") as path:
            return pd.read_csv(path)

    def getLabelsNames(self):
        return ['Normal', 'Degraded Outer Race', 'Outer Race', 'Degraded Inner Race', 'Inner Race', 'Degraded Roller Race', 'Roller Race']