import os
import bz2
import sys
import gzip
import lzma
import hashlib
import os.path
import pathlib
import tarfile
import urllib
import zipfile
from typing import IO, Any, Dict, Tuple, Callable, Optional
import requests
from tqdm import tqdm
import time

import gdown
import pandas as pd
import rarfile

try:
    from importlib.resources import path as resources_path
except ImportError:
    from pkg_resources import resource_stream as resources_path


def retry(func, tries=3, delay=2, backoff=1, **kwargs):
    for attempt in range(tries):
        try:
            return func(**kwargs)
        except Exception as e:
            if attempt < tries - 1:
                time.sleep(delay)
                delay *= backoff
            else:
                print(f"Max retries ({tries}) exceeded of the func {func.__name__} with args {str(kwargs)}")                
                raise e

def _compute_md5_dir(dir_path : str):
    # this is a fast way to check wheter there are changes in the data directory, but does not guarantee if the 
    # files content or structure has been modified
    files = sorted(os.listdir(dir_path))
    dir_desc = \
    '''
    {}
    {}
    '''.format(len(files), files)
    md5_hash = hashlib.md5()
    md5_hash.update(dir_desc.encode('utf-8'))
    md5_digest = md5_hash.hexdigest()
    return md5_digest

def _get_package_resource_dataframe(
    package: str,
    resource: str,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Compatibility function to load csv files for datasets.

    Use it instead of 'read_csv' from pandas with importlib.
    Works with Python>=3.6

    Args:
        package: package name
        resource: csv filename
        **read_csv_kwargs: extra kwargs to be passed to 'read_csv'

    Returns:
        Pandas dataframe
    """

    with resources_path(package, resource) as r:
        df = pd.read_csv(r, **read_csv_kwargs)
    return df


def calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
    # Setting the `usedforsecurity` flag does not change anything about the functionality, but indicates that we are
    # not using the MD5 checksum for cryptography. This enables its usage in restricted environments like FIPS. Without
    # it torchvision.datasets is unusable in these environments since we perform a MD5 check everywhere.
    md5 = hashlib.md5(**dict(usedforsecurity=False) if sys.version_info >= (3, 9) else dict())
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str, md5: str, **kwargs: Any) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str, md5: Optional[str] = None) -> bool:
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_file_from_google_drive(file_id: str, output: str, md5: Optional[str] = None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    #TODO: Update docs
    fpath = os.path.expanduser(output)

    if check_integrity(fpath, md5):
        print(f"Using downloaded {'and verified ' if md5 else ''}file: {fpath}")
        return

    url_base = "https://drive.google.com/uc?id="
    gdown.cached_download(url=url_base + file_id, path=fpath, md5=md5)

def download_usual_file(url: str, output_path: str, session, chunk_size=512 * 1024): # chunck size of 512kB
    # TODO: Create docs
    # based on https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
    resp = session.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
        desc=f"From: {url}\tTo: {output_path}\t",
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def download(url: str, output_path: str, session: requests.Session = None, **kwargs):
    # TODO: Create docs
    sess = session if session is not None else requests.Session()
    gdrive_file_id, _ = gdown.parse_url.parse_url(url)

    if isinstance(gdrive_file_id, bool) and not gdrive_file_id:
        retry(download_usual_file, tries=10, url=url, output_path=output_path, session=sess, **kwargs)
    else:
        # retry and session policy is already native to gdown
        download_file_from_google_drive(file_id=gdrive_file_id, output=output_path, **kwargs)

def _extract_tar(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
        tar.extractall(to_path)


def _extract_rar(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with rarfile.RarFile(from_path) as rar:
        rar.extractall(to_path)


_ZIP_COMPRESSION_MAP: Dict[str, int] = {
    ".bz2": zipfile.ZIP_BZIP2,
    ".xz": zipfile.ZIP_LZMA,
}


def _extract_zip(from_path: str, to_path: str, compression: Optional[str]) -> None:
    with zipfile.ZipFile(
        from_path,
        "r",
        compression=_ZIP_COMPRESSION_MAP[compression] if compression else zipfile.ZIP_STORED,
    ) as zip:
        zip.extractall(to_path)


_ARCHIVE_EXTRACTORS: Dict[str, Callable[[str, str, Optional[str]], None]] = {
    ".tar": _extract_tar,
    ".zip": _extract_zip,
    ".rar": _extract_rar,
}
_COMPRESSED_FILE_OPENERS: Dict[str, Callable[..., IO]] = {
    ".bz2": bz2.open,
    ".gz": gzip.open,
    ".xz": lzma.open,
}
_FILE_TYPE_ALIASES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz"),
}


def _detect_file_type(file: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Detect the archive type and/or compression of a file.
    Args:
        file (str): the filename
    Returns:
        (tuple): tuple of suffix, archive type, and compression
    Raises:
        RuntimeError: if file has no suffix or suffix is not supported
    """
    suffixes = pathlib.Path(file).suffixes
    if not suffixes:
        raise RuntimeError(
            f"File '{file}' has no suffixes that could be used to detect the archive type and compression."
        )
    suffix = suffixes[-1]

    # check if the suffix is a known alias
    if suffix in _FILE_TYPE_ALIASES:
        return (suffix, *_FILE_TYPE_ALIASES[suffix])

    # check if the suffix is an archive type
    if suffix in _ARCHIVE_EXTRACTORS:
        return suffix, suffix, None

    # check if the suffix is a compression
    if suffix in _COMPRESSED_FILE_OPENERS:
        # check for suffix hierarchy
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]

            # check if the suffix2 is an archive type
            if suffix2 in _ARCHIVE_EXTRACTORS:
                return suffix2 + suffix, suffix2, suffix

        return suffix, None, suffix

    valid_suffixes = sorted(set(_FILE_TYPE_ALIASES) | set(_ARCHIVE_EXTRACTORS) | set(_COMPRESSED_FILE_OPENERS))
    raise RuntimeError(f"Unknown compression or archive type: '{suffix}'.\nKnown suffixes are: '{valid_suffixes}'.")


def _decompress(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
    r"""Decompress a file.
    The compression is automatically detected from the file name.
    Args:
        from_path (str): Path to the file to be decompressed.
        to_path (str): Path to the decompressed file. If omitted, ``from_path`` without compression extension is used.
        remove_finished (bool): If ``True``, remove the file after the extraction.
    Returns:
        (str): Path to the decompressed file.
    """
    suffix, archive_type, compression = _detect_file_type(from_path)
    if not compression:
        raise RuntimeError(f"Couldn't detect a compression from suffix {suffix}.")

    if to_path is None:
        to_path = from_path.replace(suffix, archive_type if archive_type is not None else "")

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    compressed_file_opener = _COMPRESSED_FILE_OPENERS[compression]

    with compressed_file_opener(from_path, "rb") as rfh, open(to_path, "wb") as wfh:
        wfh.write(rfh.read())

    if remove_finished:
        os.remove(from_path)

    return to_path


def extract_archive(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
    """Extract an archive.
    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive the call is dispatched to :func:`decompress`.
    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.
    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    suffix, archive_type, compression = _detect_file_type(from_path)
    if not archive_type:
        return _decompress(
            from_path,
            os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
            remove_finished=remove_finished,
        )

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    extractor(from_path, to_path, compression)
    if remove_finished:
        os.remove(from_path)

    return to_path


def extract_archive_and_remove(from_path: str, to_path: Optional[str] = None, remove_finished: bool = False) -> str:
    """Extract an archive.

    The archive type and a possible compression is automatically detected from the file name. If the file is compressed
    but not an archive the call is dispatched to :func:`decompress`.

    Args:
        from_path (str): Path to the file to be extracted.
        to_path (str): Path to the directory the file will be extracted to. If omitted, the directory of the file is
            used.
        remove_finished (bool): If ``True``, remove the file after the extraction.

    Returns:
        (str): Path to the directory the file was extracted to.
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    suffix, archive_type, compression = _detect_file_type(from_path)
    if not archive_type:
        return _decompress(
            from_path,
            os.path.join(to_path, os.path.basename(from_path).replace(suffix, "")),
            remove_finished=remove_finished,
        )

    # We don't need to check for a missing key here, since this was already done in _detect_file_type()
    extractor = _ARCHIVE_EXTRACTORS[archive_type]

    extractor(from_path, to_path, compression)

    # Remove the zip source file
    os.remove(from_path)

    return to_path
