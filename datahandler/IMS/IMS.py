# Code made in Pycharm by Igor Varejao
from datahandler.base import RawVibrationDataset, DownloadableDataset

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
    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=CWRU_raw.resources, download_mirrors=CWRU_raw.mirrors)
        else:
            super().__init__(root_dir=root_dir, download_resources=CWRU_raw.resources, download_mirrors=None)


    def getLabelsNames(self):
        return ['Normal', 'Outer Race', 'Inner Race', 'Roller Race']