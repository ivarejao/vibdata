# Code made in Pycharm by Igor Varejao
import os

import numpy as np
import pandas as pd
import shutil

from vibdata.raw.base import DownloadableDataset, RawVibrationDataset
from vibdata.raw.utils import _get_package_resource_dataframe, extract_archive_and_remove
from vibdata.definitions import LABELS_PATH

# This dataset are composed by three tests each one describing a test-to-failure experiment.
# Each test is made by several files, representing a ~1 second record with 20Hz sample rate.
# Classification is based on the paper that can be found in https://www.sciencedirect.com/science/article/abs/pii/S0957417418303324
# ---------------------------------------------------------
#                           Test 1
# ---------------------------------------------------------
# In this test, 8 accelerometers were placed in 4 bearings in vertical and horizontal axis.
# Each file consists of 20480 lines, with 8 channels where every pair
# indicating the record of x-axis and y-axis of the bearing
# The relation between column and bearing is
# | Bearing | Colunms |
# |---------|---------|
# | 1       | 0 & 1   |
# | 2       | 2 & 3   |
# | 3       | 4 & 5   |
# | 4       | 6 & 7   |
#
# Only Bearing 3 and Bearing 4 got Faults States, therefore, are recommended to use them.
# In this test it got recorded 2156 times / files.
# | 			Bearing 3       	|
# |---------------------------------|
# | Healthy State       |   Range   |
# |---------------------|-----------|
# | Normal              | 0:1799    |
# | Degraded Inner Race | 1800:2099 |
# | Inner Race Fault    | 2100:2155 |
#
# | 			Bearing 4       	 |
# |----------------------------------|
# | Healthy State        |   Range   |
# |----------------------|-----------|
# | Normal               | 0:1399    |
# | Degraded Roller Race | 1400:1849 |
# | Roller race Fault    | 1850:2155 |
#
# ---------------------------------------------------------
#                           Test 2
# ---------------------------------------------------------
# Only Bearing 1 got Faults States, therefore, are recommended to use only it.
# In this test it got recorded 984 times / files. It got 4 channels each one for each
# bearing.
# | 			Bearing 1       	|
# |---------------------------------|
# | Healthy State       |   Range   |
# |---------------------|-----------|
# | Normal              | 0:699     |
# | Degraded Outer Race | 700:949   |
# | Outer Race Fault    | 950:983   |
#
# ---------------------------------------------------------
#                           Test 3
# ---------------------------------------------------------
# In this test no bearing got a Fault, and it got recorded 6.324 times / files


class IMS_raw(RawVibrationDataset, DownloadableDataset):

    gdrive_counterpart = {
        "filename": "IMS.zip", 
        "md5": "4d24ffef04f5869d68c0bc7cf65ebf77", 
        "id": "1r9SadjRcUkvI1wJZvi-nu9VzPyQ8oOvE"
    }
    source = ["https://data.nasa.gov/docs/legacy/IMS.zip"]
    dir_md5 = "346c8759b14b9a0a977e5c03539c2cd8"

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
    #              }

    def __init__(self, root_dir: str, download_from_source=False, with_thirdtest=False):
        super().__init__(root_dir=root_dir, download_gdrive=self.gdrive_counterpart, download_from_source=download_from_source)
        self.third_test = with_thirdtest

    def _get_test_folder(self, ntest: int) -> str:
        """
        Get the name of the folder where the files of the test will be stored
        Args:
            ntest (int): The number of the test occurred

        Returns:
            The name of the folder containing the files of test in matter
        """
        if ntest == 1:
            return "1st_test"
        elif ntest == 2:
            return "2nd_test"
        else:
            return "3rd_test"

    # Implement the abstract methods from RawVibrationalDataset
    # ---------------------------------------------------------
    def __getitem__(self, idx) -> dict:
        try:
            if not hasattr(idx, "__len__") and not isinstance(idx, slice):
                # return self.__getitem__([idx]).iloc[0]
                return self.__getitem__([idx])
            df = self.getMetaInfo()
            if isinstance(idx, slice):
                rows = df.iloc[idx.start : idx.stop : idx.step]
                range_idx = list(range(idx.start, idx.stop, idx.step))
            else:
                rows = df.iloc[idx]
                range_idx = idx
        except IndexError as error:
            print(f"{error} with idx={idx}")
            exit(1)

        signal_datas = np.empty(rows.shape[0], dtype=object)

        for i, row in enumerate(rows.itertuples()):
            path_file = os.path.join(self.raw_folder, self._get_test_folder(row.test), row.file_name)
            file_data = np.loadtxt(path_file, delimiter="\t", unpack=True)
            column = (row.bearing - 1) * 2 + (1 if row.axis == "vertical" else 0)
            signal_datas[i] = file_data[column, :]

        return {"signal": signal_datas, "metainfo": rows}

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "IMS.csv")
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels["dataset"] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            df["label"] = df["label"].apply(lambda id_label: dict_labels[id_label])
        if self.third_test:
            return df
        else:
            return df[:21184]

    def name(self):
        return "IMS"

    def download(self) -> None:
        super().download()

        # post-processing
        if self.download_from_source:
            # organize structure to follow standard
            source_dir = os.path.join(self.raw_folder, "IMS", "IMS")
            aux_dir = os.path.join(os.path.dirname(self.raw_folder), "IMS_aux")

            shutil.move(source_dir, aux_dir)
            shutil.rmtree(self.raw_folder)
            os.rename(aux_dir, self.raw_folder) 

            print("Extracting subdirectories ...")
            # extract the compressed arquives
            for test_file in filter(lambda f: f.endswith('.rar'), os.listdir(self.raw_folder)):
                extract_archive_and_remove(os.path.join(self.raw_folder, test_file))
    
            # in order to keep the pattern from the gdrive counterpart need to remove the pdf
            os.remove(os.path.join(self.raw_folder, "Readme Document for IMS Bearing Data.pdf"))
            # for some reason the 3rd test has a nested directory and the wrong name
            source_dir = os.path.join(self.raw_folder, "4th_test", "txt")
            target_dir = os.path.join(self.raw_folder, "3rd_test")
            shutil.move(source_dir, target_dir)
            shutil.rmtree(os.path.dirname(source_dir))
                