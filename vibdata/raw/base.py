import os
from abc import abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Sequence
from urllib.error import URLError

import numpy as np
import pandas as pd

from vibdata.raw.utils import extract_archive_and_remove, download_file_from_google_drive
from vibdata.definitions import LABELS_PATH


class DownloadableDataset:
    def __init__(
        self,
        root_dir: str,
        download_resources: List[Tuple[str, str]],
        download_mirrors: List = None,
        download_urls: List = None,
        extract_files=False,
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
        self.download_resources = download_resources
        self.download_mirrors = download_mirrors
        self.download_urls = download_urls
        self.extract_files = extract_files
        self.download_done = False
        if not self._check_exists():
            self.download()
            if not self._check_exists():
                raise RuntimeError("Dataset not found. You can use download=True to download it.")
        self.download_done = True

    @property
    def raw_folder(self) -> str:
        if self.download_done:
            return os.path.join(
                self.root_dir,
                self.__class__.__name__,
                self.download_resources[0][0][:-4],
            )
        else:
            return os.path.join(self.root_dir, self.__class__.__name__)

    def _check_exists(self) -> bool:
        for url, _ in self.download_resources:
            fpath = os.path.join(self.raw_folder, url[:-4])
            if not os.path.isdir(fpath):
                return False
        return True

    def download(self) -> None:
        """Download the dataset, if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        if self.download_urls is None:
            urls_list = [
                [f"{mirror}/{filename}" for filename, _ in self.download_resources] for mirror in self.download_mirrors
            ]
        else:
            if isinstance(self.download_urls[0], str):
                urls_list = [self.download_urls]
            else:
                urls_list = self.download_urls

        for i, (filename, md5) in enumerate(self.download_resources):
            for url_mirror in urls_list:
                url = url_mirror[i]
                try:
                    if self.extract_files:
                        download_file_from_google_drive(url, root=self.raw_folder, filename=filename, md5=md5)
                        extract_archive_and_remove(
                            self.raw_folder + f"/{filename}",
                            self.raw_folder + f"/{filename[:-4]}",
                        )
                    else:
                        download_file_from_google_drive(url, root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print("Failed to download:\n{}".format(error))
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))


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
