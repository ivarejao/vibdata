import os
from typing import List, Tuple
from urllib.error import URLError

from gdown import download
import shutil

import numpy as np
import pandas as pd
import requests
from vibdata.raw.base import DownloadableDataset, RawVibrationDataset
from vibdata.raw.utils import _get_package_resource_dataframe, extract_archive_and_remove
from vibdata.definitions import LABELS_PATH
import scipy

class HUST_raw(RawVibrationDataset, DownloadableDataset):

    gdrive_counterpart = {
        "filename": "HUST.zip",
        "md5": "51414ffba877ce602147e09e21b38a51",
        "id": "1G7Z7SIIvQungvvrVup7r1wy_7s5rv5Y5",
    }
    source = ["https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/cbv7jyx4p9-2.zip"]
    dir_md5 = "53cb1dfa329a493dcd357eb7f9c8be32"

    def __init__(self, root_dir: str, download_from_source=False):
        super().__init__(root_dir=root_dir, download_gdrive=self.gdrive_counterpart, download_from_source=download_from_source)


    def __getitem__(self, index : slice | int ) -> dict:
        # TODO: Pensar se vai realmenter manter o retorno como uma lista
        if isinstance(index, int):
            ret = self.__getitem__([index])
            return ret
        metainfo = self.getMetaInfo()
        if isinstance(index, slice):
            rows = metainfo.iloc[index.start : index.stop : index.step]
        else:
            rows = metainfo.iloc[index]
        
        signals = np.empty(rows.shape[0], dtype=object)
        file_names = rows["file_name"]
        for i, f_name in enumerate(file_names):
            data = scipy.io.loadmat(
                os.path.join(
                    self.raw_folder, f_name
                ),
            )
            # Remove variables native from matlab files
            data = {key : value for key, value in data.items() if not key.startswith("__")}
            data.pop("fs")  #  Remove columns that is already in the metainfo
            signal = data.pop("data")
            # TODO: Add the other infos that the sample may have like, `ru`, `rpm`
            signals[i] = signal
        
        ret = {"signal" : signals, "metainfo": rows}
        return ret
    
    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "HUST.csv")
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels["dataset"] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            df["label"] = df["label"].apply(lambda id_label: dict_labels[id_label])
        return df

    def name(self):
        return "HUST"
    
    def download(self)-> None:
        super().download()
        # post-processing
        # organize structure to follow standard
        inter_dir = "cbv7jyx4p9-2" if self.download_from_source else ""
        source_dir = os.path.join(self.raw_folder, inter_dir, "HUST bearing")
        aux_dir = os.path.join(os.path.dirname(self.raw_folder), "HUST_aux")

        shutil.move(source_dir, aux_dir)
        shutil.rmtree(self.raw_folder)
        os.rename(aux_dir, self.raw_folder)
            