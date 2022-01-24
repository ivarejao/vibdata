
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader, Dataset
import pandas as pd
from vibdata.datahandler.base import RawVibrationDataset
import hashlib
import os
import pickle
from tqdm import tqdm
import numpy as np
from vibdata.datahandler.transforms.signal import Sequential


class PickledDataset(Dataset):
    """
    This dataset loads a dataset saved by function `transform_and_saveDataset`.
    """
    def __init__(self, root_dir, transforms=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.file_names = [f for f in os.listdir(self.root_dir) if f[-4:] == '.pkl' and f != 'metainfo.pkl']
        self.file_names = sorted(self.file_names, key=lambda k: int(k[:-4]))
        with open(os.path.join(root_dir, 'metainfo.pkl'), 'rb') as f:
            self.metainfo = pickle.load(f)
        self.transforms = transforms
        assert(len(self.file_names) == len(self.metainfo['label'])), "%d %d" % (
            len(self.file_names), len(self.metainfo['label']))

    def __getitem__(self, i):
        ret = {k: v.iloc[i] if isinstance(v, pd.DataFrame) else v[i]
               for k, v in self.metainfo.items()}
        fpath = os.path.join(self.root_dir, self.file_names[i])
        with open(fpath, 'rb') as f:
            ret['signal'] = pickle.load(f)

        if(self.transforms is not None):
            if(hasattr(self.transforms, 'transform')):
                return self.transforms.transform(ret)
            return self.transforms(ret)
        return ret

    def __len__(self):
        return len(self.file_names)


def transform_and_saveDataset(dataset: RawVibrationDataset, transforms, dir_path: str, batch_size=1024) -> PickledDataset:
    """
    This function applies `transforms` to `dataset` and caches each transformed sample in a separated file in `dir_path`, 
    and finally returns a Dataset object implementing `__getitem__`.
    If this function is called with the same arguments a second time, it returns the cached dataset. 

    Args:
        dataset: Should be a `RawVibrationDataset` object implementing `__getitem__` that accepts lists of integers as parameters.
        transforms: A object (or a list of objects) implementing `__call__` or `transform`
        dir_path: path to the cache directory (Suggestion: use "/tmp" or another temporary directory)
        batch_size: 
    """
    if(not hasattr(transforms, 'transform') and not callable(transforms)):
        if(hasattr(transforms, '__iter__')):
            transforms = Sequential(*transforms)

    m = hashlib.md5()
    to_encode = [transforms, dataset.__class__.__name__, len(dataset), dataset.getMetaInfo()]
    for e in to_encode:
        m.update(str(e).encode('utf-8'))
    hash_code = m.hexdigest()
    hashfile = os.path.join(dir_path, 'hash_code')

    if(os.path.isdir(dir_path)):
        if(len(os.listdir(dir_path)) > 0):
            if(os.path.isfile(hashfile)):
                with open(hashfile, 'r') as f:
                    if(f.read().strip('\n') == hash_code):
                        return PickledDataset(dir_path)
            raise ValueError("Directory exists and it is not empty.")
    else:
        os.makedirs(dir_path)

    dataloader = DataLoader(dataset, batch_size=None, collate_fn=lambda x: x,  # do not convert to Tensor
                            sampler=BatchSampler(SequentialSampler(dataset), batch_size, False))
    metainfo_list = []
    fid = 0
    for data in tqdm(dataloader):
        if(hasattr(transforms, 'transform')):
            data_transf = transforms.transform(data)
        else:
            data_transf = transforms(data)

        for i in range(len(data_transf['signal'])):
            fpath = os.path.join(dir_path, "{}.pkl".format(fid))
            with open(fpath, 'wb') as f:
                pickle.dump(data_transf['signal'][i], f)
            fid += 1

        del data_transf['signal']
        metainfo_list.append(data_transf)

    metainfo = {}
    for k, v in metainfo_list[0].items():
        if(isinstance(v, pd.DataFrame)):
            metainfo[k] = pd.concat([m[k] for m in metainfo_list])
        else:
            metainfo[k] = np.hstack([m[k] for m in metainfo_list])

    fpath = os.path.join(dir_path, 'metainfo.pkl')
    with open(fpath, 'wb') as f:
        pickle.dump(metainfo, f)

    with open(hashfile, 'w') as f:
        f.write(hash_code)

    return PickledDataset(dir_path)
