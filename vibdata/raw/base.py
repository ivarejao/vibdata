import os
from abc import abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Sequence
from urllib.error import URLError

import numpy as np
import pandas as pd
import requests


from vibdata.raw.utils import extract_archive_and_remove, download, _ARCHIVE_EXTRACTORS, _compute_md5_dir
from vibdata.definitions import LABELS_PATH


class DownloadableDataset:
    def __init__(
        self,
        root_dir: str,
        download_gdrive: Dict[str, str],
        download_from_source : bool,
    ) -> None:
        """
        This class does not download the dataset if files are already present and their md5 hash (if available) are correct.
        Otherwise, this constructor automatically download the dataset.
        Args:
            root_dir (str): Root directory of dataset where dataset exists or where dataset will be saved when downloaded.
            download_urls: List of urls of files to download. Each element corresponds to a file. The file name will be determined by `download_resources`.
            download_mirrors: A list of urls to the "root directory" where files should be downloaded. A list of multiple urls can be provided to use multiples mirrors,
                meaning that if one url fails, the next one is used. The full url is determined by a concatenation of this url to the file_name in parameter `download_resources`.
            download_resources: List of tuples (file_name, md5sum). The file_name is a string. The md5sum can be None or a string.
            extract_files: If true, the downloaded files will be extracted (zip, tar.gz, ...).
        """

        self.root_dir = root_dir
        self.download_gdrive = download_gdrive
        self.download_from_source = download_from_source
        self.download_done = False
        if not self._check_exists():
            self.download()
            if not self._check_exists():
                # TODO: Update this error message
                raise RuntimeError("Dataset not found. You can use download=True to download it.")
        self.download_done = True

    @property
    def raw_folder(self) -> str:
        return os.path.join(
            self.root_dir,
            self.__class__.__name__,
            self.name(),
        )

    def _check_exists(self) -> bool:
        if os.path.isdir(self.raw_folder):
            # compute md5sum
            return self.dir_md5 == _compute_md5_dir(self.raw_folder)
        return False

    def download(self) -> None:
        """Download the dataset, if it doesn't exist already."""

        if self._check_exists():
            return

        if self.download_from_source:
            # download all files in self.source
            # use same session to avoid overhead
            session = requests.Session()
            os.makedirs(self.raw_folder, exist_ok=True)
            for url in self.source:
                try:
                    output_path = os.path.join(self.raw_folder, os.path.basename(url))
                    download(url, output_path, session=session, chunk_size=1024)
                    ext = os.path.basename(url)[-4:]
                    # in case is a compressed file and exist an extractor for it
                    if ext in _ARCHIVE_EXTRACTORS.keys():
                        extract_archive_and_remove(output_path, output_path[:-4])
                    print(f"Download of {url} done")
                except Exception as error:
                    print("Error downloading {}".format(url))      
                    raise error         
        else:
            # download from the google drive
            url_base = "https://drive.google.com/uc?id="
            full_url = url_base + self.download_gdrive["id"]
            
            # create all directories before `{dataset_name}` as it will be created when 
            # file is extracted
            root = os.path.dirname(self.raw_folder)
            os.makedirs(root, exist_ok=True)
            output_path = os.path.join(root, self.download_gdrive["filename"])
            try:
                download(full_url, output_path, md5=self.download_gdrive["md5"])
                extract_archive_and_remove(output_path, output_path[:-4])
            except URLError as error:
                raise RuntimeError("Error downloading {}".format(full_url))
        

class RawVibrationDataset:
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def __getitem__(self, index) -> dict:
        """
        returns:
            A dictionary with at least these two keys: 'signal', with a numpy matrix where each row is a vector of amplitudes of the signal;
                and 'metainfo', which returns the i-th row of the dataframe returned by `self.getMetaInfo`.

        """
        if hasattr(index, "__iter__"):
            sigs = []
            metainfos = []
            for i in index:
                d = self[i]
                sigs.append(d["signal"])
                row = pd.DataFrame(
                    [d["metainfo"].values],
                    columns=d["metainfo"].index.values,
                    index=[d["metainfo"].name],
                )
                metainfos.append(row)
            return {"signal": sigs, "metainfo": pd.concat(metainfos)}
        raise NotImplementedError

    def __len__(self):
        return len(self.getMetaInfo())

    @abstractmethod
    def name(self) -> str:
        """
        This should return the name of the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        """
        This does not include the time-amplitude vectors.
        Each row in returning table should refers to a single vibration signal and should have the
        same order as the signals returned by `self.__getitem__` (self[3]['metainfo']==self.getMetaInfo().iloc[3])

        The returning table should have columns "sample_rate" (in Hz) and "label".
        """
        raise NotImplementedError

    def getLabels(self, as_str=False) -> Union[List[int], List[str]]:
        df = pd.read_csv(LABELS_PATH)
        meta_labels = df.loc[df["dataset"] == self.name()]
        labels = meta_labels["label"] if as_str else meta_labels["id"]
        return labels.tolist()
