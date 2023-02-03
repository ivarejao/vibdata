import os
from vibdata.datahandler.utils import _get_package_resource_dataframe
# from FilesNames import FileNames

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
    # https://drive.google.com/file/d/1ZhIPKIn_1SrOZHnFOK69nsESFsBsEpXY/view?usp=sharing
    urls = mirrors = ["1ZhIPKIn_1SrOZHnFOK69nsESFsBsEpXY"]
    resources = [('MAFAULDA.zip', 'd3ca5a418c2ed0887d68bc3f91991f12')]

    # Data file organization
    #                  MAFAULDA.zip
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

    def __init__(self, root_dir: str, download=False):
        if (download):
            super().__init__(root_dir=root_dir, download_resources=MAFAULDA_raw.resources,
                             download_urls=MAFAULDA_raw.urls,
                             extract_files=True)
        else:
            super().__init__(root_dir=root_dir, download_resources=MAFAULDA_raw.resources)

    # Implement the abstract methods from RawVibrationalDataset
    # ---------------------------------------------------------

    def __getitem__(self, idx) -> dict:
        if (not hasattr(idx, '__len__') and not isinstance(idx, slice)):
            return self.__getitem__([idx])

        try:
            df = self.getMetaInfo()
            if (isinstance(idx, slice)):
                rows = df.iloc[idx.start: idx.stop: idx.step]
            else:
                rows = df.iloc[idx]
        except IndexError as error:
            print(f"{error} with index: {idx}")
            exit(1)

        # Get the metadata that are important to us
        file_name = rows['file_name']
        bear_name = rows['col']
        labels = rows['label']
        test_measure = rows['test_measure']

        # Create object that will store the vibedata
        signal_datas = np.empty(len(bear_name), dtype=object)

        # Begin to load the vibdata of the signals required
        for i, (f, b, t, l) in enumerate(zip(file_name, bear_name, test_measure, labels)):
            # print(f"ROW: {rows.iloc[i]}")
            # There's a subtype
            if "." in l:
                sub_labels = l.split(".")
            else:
                sub_labels = l
            # If there isn't a test measurement
            if t == "":
                raw_path = os.path.join(self.raw_folder, sub_labels, f)
            else:
                raw_path = os.path.join(self.raw_folder, *sub_labels, t, f)

            vib_data = np.loadtxt(raw_path, delimiter=',')
            signal_datas[i] = vib_data[:, b]
        signal_datas = signal_datas

        return {'signal': signal_datas, 'metainfo': rows}

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "MAFAULDA.csv",
                                             na_filter=False)
        return df

    def getLabelsNames(self):
        return ['normal', 'horizontal_misalignment', 'vertical_misalignment', 'imbalance',
                'underhang.cage_fault', 'underhang.outer_race', 'underhang.ball_fault',
                'overhang.cage_fault', 'overhang.outer_race', 'overhang.ball_fault']

    def name(self):
        return "MAFAULDA"