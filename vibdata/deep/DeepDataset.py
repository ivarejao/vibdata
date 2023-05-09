from pathlib import Path
from typing import Iterable, Union, TypedDict
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader, Dataset
import pandas as pd
from vibdata.raw.base import RawVibrationDataset
import hashlib
import os
import pickle
from tqdm import tqdm
import numpy as np
from vibdata.deep.signal.transforms import Sequential, Transform, SignalSample
from typing import Sequence, List

class DeepDataset(Dataset):
    """
    This dataset implements the methods to be used in torch framework. The data directory must be an output
    from an execution of the `convertDataset` function
    """

    def __init__(self, root_dir, transforms=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        # Load files names
        self.file_names = [f for f in os.listdir(self.root_dir) if f[-4:] == '.pkl' and f != 'metainfo.pkl']
        self.file_names = sorted(self.file_names, key=lambda k: int(k[:-4]))
        with open(os.path.join(root_dir, 'metainfo.pkl'), 'rb') as f:
            self.metainfo : pd.DataFrame = pickle.load(f)
        self.transforms = transforms
        # Confirm if there's no missing data
        assert len(self.file_names) == len(self.metainfo['label']), \
               "Number of files: %d != Labels: %d" % (len(self.file_names), len(self.metainfo['label']))


    def __getitem__(self, i : int) -> SignalSample:
        """
        Get an individual signal sample based on an integer index. If the dataset was instantiate with
        some transform, it applies these transformations into the returned signal
        Args:
            i (int): the index of the sample requeired

        Returns:
            (SignalSample) : The signals raw data (ret['signal']) and the info about it (ret['metainfo'])
        """
        ret = {'metainfo': self.metainfo.iloc[i]}
        
        fpath = os.path.join(self.root_dir, self.file_names[i])
        with open(fpath, 'rb') as f:
            # Encapsulate the signal into an array of two dimensions
            # - It needs to ensure that signal are 2d even if is a single signal
            ret['signal'] = pickle.load(f).reshape(1, -1)
        
        # Transform data if it is necessary
        if(self.transforms is not None):
            if(hasattr(self.transforms, 'transform')):
                return self.transforms.transform(ret)
            return self.transforms(ret)
        return ret

    def __len__(self) -> int:
        return len(self.file_names)


def convertDataset(dataset: RawVibrationDataset, transforms : Transform | Sequential, dir_path: Path | str, batch_size=1024):
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
    if(not hasattr(transforms, 'transform') and not callable(transforms)):
        if(hasattr(transforms, '__iter__')):
            transforms = Sequential(transforms)

    # Obscure, need to understand
    m = hashlib.md5()
    # This args must be identically to the previous execution if theres already a DeepDataset class
    # saved in the `dir_path`, therefore, must be the same transforms applied, the same version of the dataset class 
    # (Any new attribute will change the md5sum and cause an error)
    to_encode = [dataset.__class__.__name__, len(dataset), dir(dataset), transforms.__class__.__name__]
    if(hasattr(transforms, 'get_params')):
        to_encode.append(transforms.get_params())
    for e in to_encode:
        e = repr(e)
        if(' at 0x' in e):
            i = e.index(' at 0x')
            j = e[i:].index('>')
            e = e[:i]+e[i+j:]
        m.update(e.encode('utf-8'))
    hash_code = m.hexdigest()
    hashfile = os.path.join(dir_path, 'hash_code')

    # Check if an DeepDataset data is already stored in `dir_path` and, if it is, check if matches with
    # data passed, otherwise, will create the path where data will be stored  
    if(os.path.isdir(dir_path)):
        if(len(os.listdir(dir_path)) > 0):
            if(os.path.isfile(hashfile)):
                with open(hashfile, 'r') as f:
                    if(f.read().strip('\n') == hash_code):
                        return DeepDataset(dir_path)
                    else:
                        raise ValueError("Dataset corrupted! Please erase the old version.")
            raise ValueError("Directory exists and it is not empty.")
    else:
        os.makedirs(dir_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: x, # do not convert to Tensor
                            shuffle=True)
                            # sampler=BatchSampler(SequentialSampler(dataset), batch_size, False))

    metainfo_list = []
    fid = 0
    print("Transformando")
    for data in tqdm(dataloader,desc=f"Converting {dataset.name()}"):
        # Transform data
        print(len(data))
        # Iter over the batch
        for d in data:
            if(hasattr(transforms, 'transform')):
                data_transf = transforms.transform(d)
            else:
                data_transf = transforms(d)

            # Save the signal into a pickle file
            for i in range(len(data_transf['signal'])):
                fpath = os.path.join(dir_path, "{}.pkl".format(fid))
                with open(fpath, 'wb') as f:
                    pickle.dump(data_transf['signal'][i], f)
                fid += 1

            # Free memory
            del data_transf['signal']
            # Store the metainfo
            metainfo_list.append(data_transf['metainfo'])

    # Concatanate the metainfo
    metainfo = pd.concat(metainfo_list)
    # Save the metainfo
    fpath = os.path.join(dir_path, 'metainfo.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(metainfo, f)

    with open(hashfile, 'w') as f:
        f.write(hash_code)
