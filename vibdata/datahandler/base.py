from abc import abstractmethod
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import os
from torchvision.datasets.utils import download_url, download_and_extract_archive
from urllib.error import URLError


class DownloadableDataset:
    def __init__(self, root_dir: str, download_resources: List[Tuple], download_mirrors: List = None,
                 download_urls: List = None,
                 extract_files=False, concatenate_mirror_filename=True) -> None:
        self.root_dir = root_dir
        self.download_resources = download_resources
        self.download_mirrors = download_mirrors
        self.download_urls = download_urls
        self.extract_files = extract_files
        self.concatenate_mirror_filename = concatenate_mirror_filename
        if(download_mirrors is not None or download_urls is not None):
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it.')

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root_dir, self.__class__.__name__)

    def _check_exists(self) -> bool:
        for url, _ in self.download_resources:
            fpath = os.path.join(self.raw_folder, url)
            if(not os.path.isfile(fpath)):
                return False
        return True

    def download(self) -> None:
        """Download the dataset, if it doesn't exist already."""

        if(self._check_exists()):
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        if(self.download_urls is None):
            urls_list = [[f"{mirror}/{filename}" for filename, _ in self.download_resources]
                         for mirror in self.download_mirrors]
        else:
            if(isinstance(self.download_urls[0], str)):
                urls_list = [self.download_urls]
            else:
                urls_list = self.download_urls

        for i, (filename, md5) in enumerate(self.download_resources):
            for url_mirror in urls_list:
                url = url_mirror[i]
                try:
                    # print(f"Downloading {url}")
                    if(self.extract_files):
                        download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)
                    else:
                        download_url(url, root=self.raw_folder, filename=filename, md5=md5)
                except URLError as error:
                    print("Failed to download:\n{}".format(error))
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))


class RawVibrationDataset:
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self) -> Dict:
        raise NotImplementedError

    def __len__(self):
        return len(self.getLabels())

    # @abstractmethod
    # def getFreqSignals(self, ids=None):

    @abstractmethod
    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        """
        This does not include the time-amplitude vectors.
        """
        pass

    def getLabels(self, as_str=False):
        return self.getMetaInfo(labels_as_str=as_str)['label'].values

    @abstractmethod
    def getLabelsNames(self):
        raise NotImplementedError

    def getNumLabels(self) -> int:
        return len(self.getLabelsNames())

    def asSimpleForm(self) -> Dict:
        return self[:]
