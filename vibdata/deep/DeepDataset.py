import os
import copy
import pickle
import hashlib
from typing import List, Union, Iterable, Sequence, TypedDict
from pathlib import Path

import numpy as np
import pandas as pd
import numpy.typing as npt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler

from vibdata.raw.base import RawVibrationDataset
from vibdata.definitions import LABELS_PATH
from vibdata.deep.signal.transforms import Transform, Sequential, SignalSample, Filter, SequentialFilter


class DeepDataset(Dataset):
    """
    This dataset implements the methods to be used in torch framework. The data directory must be an output
    from an execution of the `convertDataset` function
    """

    def __init__(self, root_dir, transforms=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        # Load files names
        self.file_names = [f for f in sorted(os.listdir(self.root_dir)) if f[-4:] == ".pkl" and f != "metainfo.pkl"]
        self.file_names = sorted(self.file_names, key=lambda k: int(k[:-4]))
        with open(os.path.join(root_dir, "metainfo.pkl"), "rb") as f:
            self.metainfo: pd.DataFrame = pickle.load(f)
        self.transforms = transforms
        # Confirm if there's no missing data
        assert len(self.file_names) == len(self.metainfo["label"]), "Number of files: %d != Labels: %d" % (
            len(self.file_names),
            len(self.metainfo["label"]),
        )

        # Store the labels and labels_name
        self._compute_labels()

    def _compute_labels(self) -> None:
        labels = self.metainfo["label"].unique()
        labels.sort()
        self.labels = labels

        relation_labels_name = pd.read_csv(LABELS_PATH)
        dataset_labels_mask = relation_labels_name["id"].isin(self.labels)
        self.labels_name = relation_labels_name[dataset_labels_mask]["label"].unique()

    def __getitem__(self, i: int) -> SignalSample:
        """
        Get an individual signal sample based on an integer index. If the dataset was instantiate with
        some transform, it applies these transformations into the returned signal
        Args:
            i (int): the index of the sample requeired

        Returns:
            (SignalSample) : The signals raw data (ret['signal']) and the info about it (ret['metainfo'])
        """
        ret = {"metainfo": self.metainfo.iloc[i]}

        fpath = os.path.join(self.root_dir, self.file_names[i])
        with open(fpath, "rb") as f:
            # Encapsulate the signal into an array with sample select axis
            # - It needs to ensure that signal are at least 2d even if is a single signal
            signal = pickle.load(f)
            ret["signal"] = np.expand_dims(signal, axis=0)

        # Transform data if it is necessary
        if self.transforms is not None:
            if hasattr(self.transforms, "transform"):
                return self.transforms.transform(ret)
            return self.transforms(ret)
        return ret

    def __len__(self) -> int:
        return len(self.file_names)

    def get_labels(self) -> npt.NDArray[np.int_]:
        return self.labels

    def get_labels_name(self) -> npt.NDArray[np.str_]:
        return self.labels_name

    # Getters and setters
    def get_metainfo(self) -> pd.DataFrame:
        return self.metainfo

    def set_metainfo(self, metainfo: pd.DataFrame) -> None:
        self.metainfo = metainfo

    def get_file_names(self) -> List[str]:
        return self.file_names

    def set_file_names(self, file_names: List[str]) -> None:
        self.file_names = file_names

    def __deepcopy__(self, memo) -> "DeepDataset":
        """
        Create a deep copy of the DeepDataset object.

        Parameters:
        - memo: A dictionary used by the deepcopy function to track already copied objects.

        Returns:
        A new instance of DeepDataset with the same root_dir, transforms, and copied attributes.
        """
        # Create a new instance of DeepDataset with the same root_dir and transforms
        new_dataset = DeepDataset(self.root_dir, self.transforms)

        # Create new copies of the mutable attributes
        new_dataset.file_names = copy.deepcopy(self.file_names, memo)
        new_dataset.metainfo = self.metainfo.copy(deep=True)
        new_dataset.labels_name = copy.deepcopy(self.labels_name, memo)
        new_dataset.labels = copy.deepcopy(self.labels, memo)

        return new_dataset


def resample_dataset(dataset: DeepDataset, indexes: np.ndarray) -> DeepDataset:
    """
    Resamples the given dataset by selecting the samples at the specified indexes.

    Args:
        dataset (DeepDataset): The dataset to be resampled.
        indexes (np.ndarray): The indexes of the samples to be selected.

    Returns:
        DeepDataset: The resampled dataset.
    """
    new_dataset = copy.deepcopy(dataset)
    # Get the file_names and convert to numpy array so it the indexing can be done
    current_file_names = np.array(new_dataset.get_file_names())
    new_dataset.set_file_names(current_file_names[indexes].tolist())
    # Update the metainfo with only the indexes passed
    new_dataset.set_metainfo(new_dataset.metainfo.iloc[indexes])
    return new_dataset


def convertDataset(
    dataset: RawVibrationDataset,
    transforms: Transform | Sequential,
    dir_path: Path | str,
    filter: Filter | SequentialFilter = None,
    batch_size=32,
):
    """
    This function applies `transforms` to `dataset` and caches each transformed sample in a separated file in `dir_path`,
    and finally returns a Dataset object implementing `__getitem__`.
    If this function is called with the same arguments a second time, it returns the cached dataset.

    Args:
        dataset: Should be an iterable object implementing `__len__` and `__getitem__`.
            The `__getitem__` method should accept lists of integers as parameters.
        transforms: A object (or a list of objects) implementing `__call__` or `transform`
        dir_path: path to the cache directory (Suggestion: use "/tmp" or another temporary directory)
        batch_size:
    """
    if not hasattr(transforms, "transform") and not callable(transforms):
        if hasattr(transforms, "__iter__"):
            transforms = Sequential(transforms)

    # Obscure, need to understand
    m = hashlib.md5()
    # This args must be identically to the previous execution if theres already a DeepDataset class
    # saved in the `dir_path`, therefore, must be the same transforms applied, the same version of the dataset class
    # (Any new attribute will change the md5sum and cause an error)
    to_encode = [
        dataset.__class__.__name__,
        len(dataset),
        dir(dataset),
        transforms.__class__.__name__,
    ]
    if hasattr(transforms, "get_params"):
        to_encode.append(transforms.get_params())
    for e in to_encode:
        e = repr(e)
        if " at 0x" in e:
            i = e.index(" at 0x")
            j = e[i:].index(">")
            e = e[:i] + e[i + j :]
        m.update(e.encode("utf-8"))
    hash_code = m.hexdigest()
    hashfile = os.path.join(dir_path, "hash_code")

    # Check if an DeepDataset data is already stored in `dir_path` and, if it is, check if matches with
    # data passed, otherwise, will create the path where data will be stored
    if os.path.isdir(dir_path):
        if len(os.listdir(dir_path)) > 0:
            if os.path.isfile(hashfile):
                with open(hashfile, "r") as f:
                    if f.read().strip("\n") == hash_code:
                        return DeepDataset(dir_path)
                    else:
                        raise ValueError("Dataset corrupted! Please erase the old version.")
            raise ValueError("Directory exists and it is not empty.")
    else:
        os.makedirs(dir_path)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda x: x,  # do not convert to Tensor
        shuffle=False,
    )
    # sampler=BatchSampler(SequentialSampler(dataset), batch_size, False))

    metainfo_list = []
    fid = 0
    print("Transformando")
    for data in tqdm(dataloader, desc=f"Converting {dataset.name()}"):
        # Transform data
        # Iter over the batch
        for d in data:  # 'd' is a batch of one sample
            # Apply filter
            if filter:
                d = filter.filter(d)
                if d is None:
                    continue

            if hasattr(transforms, "transform"):
                data_transf = transforms.transform(d)
            else:
                data_transf = transforms(d)
            d = np.squeeze(d, axis=0)

            # Save the signal into a pickle file without the sample-select axis
            for i in range(len(data_transf["signal"])):
                fpath = os.path.join(dir_path, "{:07d}.pkl".format(fid))
                with open(fpath, "wb") as f:
                    pickle.dump(data_transf["signal"][i], f)
                fid += 1

            # Free memory
            del data_transf["signal"]
            # Store the metainfo
            metainfo_list.append(data_transf["metainfo"])

    # Concatanate the metainfo
    if len(metainfo_list) > 0:
        metainfo = pd.concat(metainfo_list)
    else:
        print("No samples were saved, please check filters, returning None")
        return None
    # Save the metainfo
    fpath = os.path.join(dir_path, "metainfo.pkl")
    with open(fpath, "wb") as f:
        pickle.dump(metainfo, f)

    with open(hashfile, "w") as f:
        f.write(hash_code)

    return DeepDataset(dir_path)
