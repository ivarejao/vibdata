import os
from importlib import resources
#from FilesNames import FileNames

import numpy as np
import pandas as pd

from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset

class MAFAULDA_raw(RawVibrationDataset, DownloadableDataset):

    source = "http://www02.smt.ufrj.br/~offshore/mfs/database/mafaulda/full.zip"
    """
    DATAFILE_NAMES = {
        "horizontal-misalignment" : FileNames.horizontal_misalignment,
        "imbalance" : FileNames.imbalance,
        "normal" : FileNames.normal,
        "overhang" : FileNames.overhang,
        "underhang" : FileNames.underhang,
        "vertical-misalignment" : FileNames.vertical_misalignment
    }
    """
    #https://drive.google.com/file/d/1ZhIPKIn_1SrOZHnFOK69nsESFsBsEpXY/view?usp=sharing
    mirrors = ["1ZhIPKIn_1SrOZHnFOK69nsESFsBsEpXY"]
    resources = [('full.zip', 'd3ca5a418c2ed0887d68bc3f91991f12')]


    # Data file organization
    #                  full.zip
    #                    |
    #      ----------------------------
    #     |             |             |
    #   [--------types of labels--------]
    #    |             |             |
    #   ...           ...           ...
    #
    #
    # Some labels will have inner directories specifing some kind of info about the sample.
    # E.g.
    #           imbalance -> types of loads [6g, 10g, ...]
    #
    # For organization purpose we created a class that stores the file names in a different file


    # TODO: Find a way to extract the the rar packates of each test


    def __init__(self, root_dir: str, download=False):
        if(download):
            super().__init__(root_dir=root_dir, download_resources=MAFAULDA_raw.resources, download_urls=MAFAULDA_raw.mirrors,
                                extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=MAFAULDA_raw.resources)
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
        bear_name = rows['bearing']
        signal_datas = np.empty(len(bear_name), dtype=object)

        for i, (f, b) in enumerate(zip(file_name, bear_name)):
            file_data = np.gentext(os.path.join(self.raw_folder, f), delimiter=',')
            signal_datas[i] = file_data[:, FirstTest.back_bearing(b)]
        signal_datas = signal_datas

        return {'signal': signal_datas, 'metainfo': rows}

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        with resources.path(__package__, "IMS.csv") as path:
            return pd.read_csv(path)

    def getLabelsNames(self):
        return ['normal', 'horizontal_misalignment', 'vertical_misalignment', 'imbalance',
                'underhang.cage_fault', 'underhang.outer_race', 'underhang.ball_fault',
                'overhang.cage_fault', 'overhang.outer_race', 'overhang.ball_fault']
