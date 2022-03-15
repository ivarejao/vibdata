from abc import abstractmethod
from typing import Dict, List, Sequence, Tuple, Optional
import pandas as pd
import numpy as np
import os
from torchvision.datasets.utils import download_url, extract_archive
from urllib.error import URLError

class DownloadableDataset:
    def __init__(self, root_dir: str, download_resources: List[Tuple[str, str]], download_mirrors: List = None,
                 download_urls: List = None,
                 extract_files=False) -> None:
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
                raise RuntimeError('Dataset not found. You can use download=True to download it.')
        self.download_done = True

    @property
    def raw_folder(self) -> str:
        if self.download_done:
            return os.path.join(self.root_dir, self.__class__.__name__, self.download_resources[0][0][:-4])
        else:
            return os.path.join(self.root_dir, self.__class__.__name__)

    def _check_exists(self) -> bool:
        for url, _ in self.download_resources:
            fpath = os.path.join(self.raw_folder, url[:-4])
            if(not os.path.isdir(fpath)):
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
                        download_file_from_google_drive(url, root=self.raw_folder, filename=filename, md5=md5)
                        extract_archive(self.raw_folder + f'/{filename}', self.raw_folder + f'/{filename[:-4]}')
                        os.remove(self.raw_folder + f'/{filename}')
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
        if(hasattr(index, '__iter__')):
            sigs = []
            metainfos = []
            for i in index:
                d = self[i]
                sigs.append(d['signal'])
                row = pd.DataFrame([d['metainfo'].values], columns=d['metainfo'].index.values,
                                   index=[d['metainfo'].name])
                metainfos.append(row)
            return {'signal': sigs, 'metainfo': pd.concat(metainfos)}
        raise NotImplementedError

    def __len__(self):
        return len(self.getLabels())

    @abstractmethod
    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        """
        This does not include the time-amplitude vectors.
        Each row in returning table should refers to a single vibration signal and should have the 
        same order as the signals returned by `self.__getitem__` (self[3]['metainfo']==self.getMetaInfo().iloc[3])

        The returning table should have columns "sample_rate" (in Hz) and "label".
        """
        raise NotImplementedError

    def getLabels(self, as_str=False):
        return self.getMetaInfo(labels_as_str=as_str)['label'].values

    @abstractmethod
    def getLabelsNames(self) -> Sequence[str]:
        raise NotImplementedError

    def getNumLabels(self) -> int:
        return len(self.getLabelsNames())

def download_file_from_google_drive(file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    import itertools
    import torchvision.datasets.utils as utils
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and utils.check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = utils._get_confirm_token(response)
        max_iter = 3
        while ("application/" not in response.headers['Content-Type']):
            max_iter -= 1
            response = session.get(url, params={'id': file_id, 'confirm': True}, stream=True)
            if max_iter <= 0:
                raise GetsExceeded(response.headers['Content-Type'])

        # Ideally, one would use response.status_code to check for quota limits, but google drive is not consistent
        # with their own API, refer https://github.com/pytorch/vision/issues/2992#issuecomment-730614517.
        # Should this be fixed at some place in future, one could refactor the following to no longer rely on decoding
        # the first_chunk of the payload
        response_content_generator = response.iter_content(32768)
        first_chunk = None
        while not first_chunk:  # filter out keep-alive new chunks
            first_chunk = next(response_content_generator)

        if utils._quota_exceeded(first_chunk):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        utils._save_response_content(itertools.chain((first_chunk, ), response_content_generator), fpath)
        response.close()


class GetsExceeded(BaseException):

    def __init__(self, ctype, message="Gets Exceeded, with the last one returning a content-type: "):
        self.contentType = ctype
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message} -> {self.contentType}'