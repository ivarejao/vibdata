import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import requests
from bs4 import BeautifulSoup
import re

from vibdata.raw.base import DownloadableDataset, RawVibrationDataset
from vibdata.raw.utils import _get_package_resource_dataframe
from vibdata.definitions import LABELS_PATH


class PU_raw(RawVibrationDataset, DownloadableDataset):
    """
    Data source: https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/
    LICENSE: Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) [https://creativecommons.org/licenses/by-nc/4.0/]
    """

    # mirrors = ["http://groups.uni-paderborn.de/kat/BearingDataCenter"]
    gdrive_counterpart = {
        "filename": "PU.zip",
        "md5": "1beb53c6fb79436895787e094a22302f",
        "id": "1PZLt3h1x_rjY6EfWV3yNl3FSDWH4o-Ii"
    }
    source = ['https://groups.uni-paderborn.de/kat/BearingDataCenter/']
    dir_md5 = "2ae4fbc4c9cc3601b5ffc30b31364309"

    def __init__(self, root_dir: str, download_from_source=False):
        super().__init__(root_dir=root_dir, download_gdrive=self.gdrive_counterpart, download_from_source=download_from_source)

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        df = _get_package_resource_dataframe(__package__, "PU.csv")
        if labels_as_str:
            # Create a dict with the relation between the centralized label with the actually label name
            all_labels = pd.read_csv(LABELS_PATH)
            dataset_labels: pd.DataFrame = all_labels.loc[all_labels["dataset"] == self.name()]
            dict_labels = {id_label: labels_name for id_label, labels_name, _ in dataset_labels.itertuples(index=False)}
            df["label"] = df["label"].apply(lambda id_label: dict_labels[id_label])
        return df

    def __getitem__(self, i) -> dict:
        if not hasattr(i, "__len__") and not isinstance(i, slice):
            return self.__getitem__([i])

        if isinstance(i, slice):
            range_idx = list(range(i.start, i.stop, i.step))
            data_i = self.getMetaInfo().iloc[i]
            fname, bearing_code = data_i["file_name"], data_i["bearing_code"]
            signal_datas = np.empty(len(range_idx), dtype=object)
            for j in range(len(range_idx)):
                full_fname = os.path.join(self.raw_folder, bearing_code.iloc[j], fname.iloc[j])
                data = loadmat(full_fname, simplify_cells=True)[fname.iloc[j].split(".")[0]]
                signal_datas[j] = PU_raw._getVibration_1(data)

            return {"signal": signal_datas, "metainfo": data_i}

        data_i = self.getMetaInfo().iloc[i]
        fname, bearing_code = data_i["file_name"], data_i["bearing_code"]

        if isinstance(i, list):
            signal_datas = np.empty(len(i), dtype=object)
            for j in range(len(i)):
                full_fname = os.path.join(self.raw_folder, bearing_code.iloc[j], fname.iloc[j])
                data = loadmat(full_fname, simplify_cells=True)[fname.iloc[j].split(".")[0]]
                signal_datas[j] = PU_raw._getVibration_1(data)
            return {"signal": signal_datas, "metainfo": data_i}

        full_fname = os.path.join(self.raw_folder, bearing_code, fname)
        data = loadmat(full_fname, simplify_cells=True)[fname.split(".")[0]]
        sig = PU_raw._getVibration_1(data)
        return {"signal": sig, "metainfo": data_i}

    @staticmethod
    def _getVibration_1(data):
        Ys = data["Y"]
        for y in Ys:
            if y["Name"] == "vibration_1":
                return y["Data"]
        raise ValueError

    def asSimpleForm(self):
        metainfo = self.getMetaInfo()
        sigs = []
        files_info = metainfo[["file_name", "bearing_code"]]
        for _, (f, b) in tqdm(files_info.iterrows(), total=len(files_info)):
            full_fname = os.path.join(self.raw_folder, b, f)
            data = loadmat(full_fname, simplify_cells=True)[f.split(".")[0]]
            sigs.append(PU_raw._getVibration_1(data))
        return {"signal": sigs, "metainfo": metainfo}

    def name(self):
        return "PU"

    def download(self):
        # pre-process
        if self.download_from_source:
            # fetch all urls from the files stored at the index source site
            files_endpoints = []
            source_url = self.source[0]
            try:
                response = requests.get(source_url)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')
                table = soup.find('table')
                # extract all href attributes from anchor tags in the table
                hrefs = [link['href'] for link in table.find_all('a', href=True)]
                r = re.compile("K.*rar")
                matches = list(filter(r.match, hrefs))
                matches = list(map(lambda f: os.path.join(source_url, f), matches))
                files_endpoints.extend(matches)
                
            except requests.HTTPError as e:
                print(f"Failed to download the {self.name} Dataset from the source. Please ensure the endpoint is working.\n{str(e)}")
                raise e

            # update the self.source with the actually files urls
            self.source = files_endpoints

        super().download()
