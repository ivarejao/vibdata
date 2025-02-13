import os
from typing import List, Tuple
from urllib.error import URLError

from gdown import download

import numpy as np
import pandas as pd
import requests
from vibdata.raw.base import DownloadableDataset, RawVibrationDataset
from vibdata.raw.utils import _get_package_resource_dataframe, extract_archive_and_remove
from vibdata.definitions import LABELS_PATH
import scipy

class UORED_raw(RawVibrationDataset, DownloadableDataset):

    mirrors = [""]
    resources = [("UORED.zip", "9c6e86e4dc2f741a4c3f016fd5ec93d0")]
    source = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/y2px5tg92h-4.zip"

    def __init__(self, root_dir: str, download : bool = False) -> None:
        if download:
            super().__init__(
                root_dir=root_dir,
                download_resources=UORED_raw.resources,
                download_urls=UORED_raw.mirrors,
                extract_files=True,
            )
        else:
            super().__init__(root_dir=root_dir, download_resources=UORED_raw.resources)

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
            tag_name = os.path.basename(f_name).replace(".mat", "")
            signal = data[tag_name][:, 0] # The accelerometer data
            # TODO: Add the other infos that the sample may have like, `temperature` and `acoustic`
            signals[i] = signal
        
        ret = {"signal" : signals, "metainfo": rows}
        return ret

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "UORED.csv")
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels["dataset"] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            df["label"] = df["label"].apply(lambda id_label: dict_labels[id_label])
        return df

    def name(self):
        return "UORED"
    

    def download(self) -> None:
        """
        Override the download method, applying the download directly from the source instead of the google drive
        """
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        try:
            zip_path = os.path.join(self.raw_folder, self.name() + ".zip")
            download(self.source, output=zip_path)
            extract_archive_and_remove(zip_path, self.raw_folder)
            # Rename directory to match the default pattern
            os.rename(
                os.path.join(self.raw_folder, "University of Ottawa Rolling-element Dataset – Vibration and Acoustic Faults under Constant Load and Speed conditions (UORED-VAFCLS)"), 
                os.path.join(self.raw_folder, self.name())
            )
        except URLError as error:
            print("Failed to download:\n{}".format(error))
        finally:
            print()