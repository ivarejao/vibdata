import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat

from vibdata.raw.base import DownloadableDataset, RawVibrationDataset
from vibdata.raw.utils import _get_package_resource_dataframe
from vibdata.definitions import LABELS_PATH


class UOC_raw(RawVibrationDataset, DownloadableDataset):
    """
    Data source: https://figshare.com/articles/dataset/Gear_Fault_Data/6127874/1
    LICENSE: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) [https://creativecommons.org/licenses/by-nc/4.0/]
    """

    gdrive_counterpart = {
        "filename": "UOC.zip",
        "md5": "c33f1f6117ee4913257086007790df35",
        "id": "1oJHir0Faq_kgFnPPMaLSVyBJb6szjEOL",
    }
    source = ["https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/11053469/DataForClassification_TimeDomain.mat"]
    dir_md5 = "1f1db45f476512c2c17eb60586d71ea6"

    def __init__(self, root_dir: str, download_from_source=False):
        super().__init__(root_dir=root_dir, download_gdrive=self.gdrive_counterpart, download_from_source=download_from_source)

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "UOC.csv")
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels["dataset"] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            df["label"] = df["label"].apply(lambda id_label: dict_labels[id_label])
        return df

    def __getitem__(self, i) -> pd.DataFrame:
        if not hasattr(i, "__len__") and not isinstance(i, slice):
            return self.__getitem__([i])
        df = self.getMetaInfo()
        if isinstance(i, slice):
            rows = df.iloc[i.start : i.stop : i.step]
        else:
            rows = df.iloc[i]

        file_name = rows["file_name"]
        position = rows["position"]

        signal_datas = np.empty(len(file_name), dtype=object)
        full_fname = os.path.join(self.raw_folder, file_name.iloc[0])
        data = loadmat(full_fname, simplify_cells=True)["AccTimeDomain"]

        for i, (f, p) in enumerate(zip(file_name, position)):
            signal_datas[i] = data[:, p]
        signal_datas = signal_datas

        return {"signal": signal_datas, "metainfo": rows}

    def asSimpleForm(self):
        metainfo = self.getMetaInfo()
        sigs = []
        file_info = ["DataForClassification_TimeDomain.mat", "AccTimeDomain"]
        full_fname = os.path.join(self.raw_folder, file_info[0])
        sigs = loadmat(full_fname, simplify_cells=True)[file_info[1]]
        return {"signal": sigs, "metainfo": metainfo}

    def name(self):
        return "UOC"
