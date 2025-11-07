import os
from typing import Dict, Union
from urllib.error import URLError
from urllib.request import urlretrieve, Request, urlopen
import requests
import time

import numpy as np
import pandas as pd
from scipy.io import loadmat

from bs4 import BeautifulSoup
import re


from vibdata.raw.base import DownloadableDataset, RawVibrationDataset
from vibdata.raw.utils import _get_package_resource_dataframe, check_integrity
from vibdata.definitions import LABELS_PATH


class CWRU_raw(RawVibrationDataset, DownloadableDataset):

    # mirrors = ["https://engineering.case.edu/sites/default/files"]
    # resources = [(name, md5_value) for name, md5_value in zip(ALLNAMES, MD5SUMS)]
    # https://drive.google.com/file/d/1G2vfms1QDlkdzqL_LAQdMIQAoludxBNj/view?usp=sharing
    gdrive_counterpart = {
        "filename": "CWRU.zip",
        "md5": "d7d3042161080fc82e99d78464fa2914",
        "id": "1G2vfms1QDlkdzqL_LAQdMIQAoludxBNj",
    }
    source = [
        "https://engineering.case.edu/bearingdatacenter/normal-baseline-data",
        "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data",
        "https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data",
        "https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data",
        "https://engineering.case.edu/bearingdatacenter/12k-fan-end-bearing-fault-data"
    ]
    dir_md5 = "b8c2b518389a7a345a6c0a1fa672aff5"

    def __init__(self, root_dir: str, download_from_source=False):
        super().__init__(root_dir=root_dir, download_gdrive=self.gdrive_counterpart, download_from_source=download_from_source)

    def __getitem__(self, i) -> dict:
        if not hasattr(i, "__len__") and not isinstance(i, slice):
            ret = self.__getitem__([i])
            return ret
        df = self.getMetaInfo()
        if isinstance(i, slice):
            rows = df.iloc[i.start : i.stop : i.step]
        else:
            rows = df.iloc[i]
        file_name = rows["file_name"]
        var_name = rows["variable_name"]
        signal_datas = np.empty(len(var_name), dtype=object)
        for i, (f, v) in enumerate(zip(file_name, var_name)):
            data = loadmat(
                os.path.join(self.raw_folder, f),
                simplify_cells=True,
                variable_names=[v],
            )
            signal_datas[i] = data[v]
        signal_datas = signal_datas
        return {"signal": signal_datas, "metainfo": rows}

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "CWRU.csv")
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels["dataset"] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            df["label"] = df["label"].apply(lambda id_label: dict_labels[id_label])
        return df

    def download(self) -> None:
        """
        Override the download method, applying the download directly from the source instead of the google drive
        """
        # Fetch the urls from each file if download from source
        if self.download_from_source:
            files_endpoints = []

            for src in self.source:
                try:
                    response = requests.get(src)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, 'html.parser')
                    table = soup.find('table')
                    matches = re.findall("https://.*mat", str(table))
                    files_endpoints.extend(matches)
                    
                except requests.HTTPError as e:
                    print(f"Failed to download the {self.name} Dataset from the source. Please ensure the endpoint is working.\n{str(e)}")
                    raise e

            # update the self.source with the actually files urls
            self.source = files_endpoints
        super().download()

    def name(self):
        return "CWRU"
