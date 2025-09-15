import os
from typing import Union, Sequence

import numpy as np
import pandas as pd
import requests
from scipy.io import loadmat

from vibdata.raw.base import DownloadableDataset, RawVibrationDataset
from vibdata.raw.utils import extract_archive_and_remove, _get_package_resource_dataframe
from vibdata.definitions import LABELS_PATH
import shutil


class MFPT_raw(RawVibrationDataset, DownloadableDataset):
    """
    FIXME: metainfo
    """

    # mirrors = ["https://www.mfpt.org/wp-content/uploads/2020/02/"]
    gdrive_counterpart = {
        "filename": "MFPT.zip",
        "md5": "4631f552c6a0769996ee1d09e5feb209",
        "id": "1VxGlOMCEED7jy2qAoE9nKYAK8h5i6TIb",
    }
    source = ["https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"]
    dir_md5 = "5ca1781103c2c7e892556d3611193d1c"

    def __init__(self, root_dir: str, download_from_source=False):
        super().__init__(root_dir=root_dir, download_gdrive=self.gdrive_counterpart, download_from_source=download_from_source)

    def __getitem__(self, i) -> dict:
        if not hasattr(i, "__len__") and not isinstance(i, slice):
            ret = self.__getitem__([i])
            # ret['signal'] = ret['signal'][i]
            # ret['metainfo'] = ret['metainfo'].iloc[i]
            return ret
        df = self.getMetaInfo()
        if isinstance(i, slice):
            rows = df.iloc[i.start : i.stop : i.step]
        else:
            rows = df.iloc[i]
        file_name = rows["file_name"]
        signal_datas = np.empty(len(file_name), dtype=object)
        for i, f in enumerate(file_name):
            data = loadmat(
                os.path.join(self.raw_folder, f),
                simplify_cells=True,
                variable_names=["bearing"],
            )
            signal_datas[i] = data["bearing"]["gs"]
        signal_datas = signal_datas

        return {"signal": signal_datas, "metainfo": rows}

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "MFPT.csv")
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels["dataset"] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            df["label"] = df["label"].apply(lambda id_label: dict_labels[id_label])
        return df

    def name(self):
        return "MFPT"

    def download(self) -> None:
        super().download()
        # post processing when downloading from source
        if self.download_from_source:
            # organizing directory in the correct structure
            source_dir = os.path.join(self.raw_folder, "MFPT-Fault-Data-Sets-20200227T131140Z-001/MFPT Fault Data Sets")
            aux_dir = os.path.join(os.path.dirname(self.raw_folder), "MFPT_aux")
            target_dir = os.path.join(os.path.dirname(self.raw_folder), "MFPT")
            
            shutil.move(source_dir, aux_dir)
            shutil.rmtree(target_dir)
            os.rename(aux_dir, target_dir)