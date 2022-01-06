
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader, Dataset
import pandas as pd
from vibdata.datahandler.base import RawVibrationDataset
import hashlib
import os
import pickle
from tqdm import tqdm


def transform_and_saveDataset(dataset: RawVibrationDataset, transforms, dir_path: str):
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
        os.mkdir(dir_path)

    dataloader = DataLoader(dataset, batch_size=None,
                            sampler=BatchSampler(SequentialSampler(dataset), 32, False))
    fid = 0
    for data in tqdm(dataloader):
        if(hasattr(transforms, 'transform')):
            data_transf = transforms.transform(data)
        else:
            data_transf = transforms(data)

        for i in range(len(data_transf['signal'])):
            data_i = {k: V.iloc[i] if(isinstance(V, pd.DataFrame)) else V[i]
                      for k, V in data_transf.items()}
            fpath = os.path.join(dir_path, "{}.pkl".format(fid))
            with open(fpath, 'wb') as f:
                pickle.dump(data_i, f)
            fid += 1

    with open(hashfile, 'w') as f:
        f.write(hash_code)

    return PickledDataset(dir_path)


class PickledDataset(Dataset):
    def __init__(self, root_dir) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.file_names = [f for f in os.listdir(self.root_dir) if f[-4:] == '.pkl']

    def __getitem__(self, i):
        fpath = os.path.join(self.root_dir, self.file_names[i])
        with open(fpath, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.file_names)

# class TransformDataset(Dataset):
#     def __init__(self, dataset: RawVibrationDataset, transforms) -> None:
#         super().__init__()
#         self.dataset = dataset
#         self.transforms = transforms

#     def __iter__(self):
#         for data in self.dataset:
#             yield self.transforms(data)
