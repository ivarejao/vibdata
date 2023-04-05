# Code made in Pycharm by Igor Varejao
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Union

from torch.utils.data import DataLoader
from vibdata.datahandler.DeepDataset import convertDataset, DeepDataset
import vibdata.datahandler as datahandler
from vibdata.datahandler.base import RawVibrationDataset
from vibdata.datahandler.transforms.signal import Sequential, SplitSampleRate, Split, NormalizeSampleRate

import warnings
warnings.filterwarnings('ignore')

class DictConcurrency():

    def __init__(self):
        self.lock = threading.Lock()
        self.dict = {}

    def add(self, key, element):
        self.lock.acquire()
        try:
            self.dict[key] = element
        finally:
            self.lock.release()


def worker(datasets: DictConcurrency, dataset_class, name):
    dt = dataset_class('/tmp/', download=True)  # Instantiate
    datasets.add(name, dt)


def load_datasets(names: List[str]) -> Dict[str, RawVibrationDataset]:
    """
    Load all datasets with concurrency
    """
    all_datasets = {
        "CWRU": datahandler.CWRU_raw,
        "EAS": datahandler.EAS_raw,
        "IMS": datahandler.IMS_raw,
        "MAFAULDA": datahandler.MAFAULDA_raw,
        "MFPT": datahandler.MFPT_raw,
        "PU": datahandler.PU_raw,
        "RPDBCS": datahandler.RPDBCS_raw,
        "UOC": datahandler.UOC_raw,
        "XJTU": datahandler.XJTU_raw,
        "SEU": datahandler.SEU_raw,
    }

    selected_datasets = {dt_name: all_datasets[dt_name] for dt_name in names}

    # Instantiate each dataset
    starttime = time.time()
    print("Loading datasets")
    threads = []
    result = DictConcurrency()
    for name, dt in selected_datasets.items():
        t = threading.Thread(target=worker, args=(result, dt, name), name=name)
        threads.append(t)
        t.start()

    # Join threads
    for t in threads:
        t.join()
    print('That took {} seconds'.format(time.time() - starttime))

    return result.dict

def deep_worker(raw_dataset : RawVibrationDataset, root_path : Path):
    """
    Convert the dataset and save it on root_path
    """
    convertDataset(raw_dataset, Sequential([]), root_path, batch_size=2)

def convert_datasets(raw_datasets: Dict[str, RawVibrationDataset], root_dir : Path):

    threads = []

    # Begin the process
    starttime = time.time()
    print("# Converting raw datasets into deep datasets")

    # Create threads to convert the datasets
    for name, raw_dt in raw_datasets.items():
        dataset_path = root_dir / name
        thr = threading.Thread(target=deep_worker, args=(raw_dt, dataset_path), name=name)
        threads.append(thr)
        thr.start()

    # Join all the opened threads into the main
    for thr in threads:
        thr.join()
    print('-> converting took {} seconds'.format(time.time() - starttime))



def instantiate_deep_datasets(root_dir : Path, transforms : Union[Sequential, None], names : List[str]) -> List[DeepDataset]:
    stored_datasets = [root_dir / dataset_names for dataset_names in names]

    deep_datasets = []
    # Do not need concurrency
    for dt_path in stored_datasets:
        deep_datasets.append(
            DeepDataset(dt_path, transforms)
        )

    return deep_datasets



def main():
    # Datasets used
    # datasets_names = ['CWRU', 'IMS', 'MAFAULDA', 'MFPT', 'UOC', 'XJTU']
    datasets_names = ['CWRU']
    raw_datasets = load_datasets(datasets_names)


    root_dir = Path('/home/igor/Desktop/NINFA/datasets/')
    biggest_sample_rate = 97656  # MFPT
    transforms = Sequential([SplitSampleRate(), NormalizeSampleRate(biggest_sample_rate)])
    convert_datasets(raw_datasets, root_dir)
    deep_datasets = instantiate_deep_datasets(root_dir, transforms, datasets_names)
    sizes = set()
    for name, dataset in zip(datasets_names, deep_datasets):
        loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x:x)
        total = 0
        from functools import reduce
        for ret in loader:
            desencapsulate = lambda x : x if isinstance(x, int) else x['metainfo'].shape[0]
            get_total = lambda x, y : desencapsulate(x) + desencapsulate(y)
            total += reduce(get_total, ret)
            # Add sample_size
            for sig in ret:
                sizes.add(sig['signal'].shape[1])
        print(f"[{name}] Total samples : {total}")
        print(f"Num: {sizes}")

if __name__ == "__main__":
    main()
