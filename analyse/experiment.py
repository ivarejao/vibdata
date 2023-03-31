# Code made in Pycharm by Igor Varejao
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Union

import torch
from vibdata.datahandler.DeepDataset import convertDataset, DeepDataset
import vibdata.datahandler as datahandler
from vibdata.datahandler.base import RawVibrationDataset
from vibdata.datahandler.transforms.signal import Sequential, SplitSampleRate, Split


class ListConcurrency():

    def __init__(self):
        self.lock = threading.Lock()
        self.list = []

    def add(self, element):
        self.lock.acquire()
        try:
            self.list.append(element)
        finally:
            self.lock.release()


def worker(datasets: ListConcurrency, dataset_class, name):
    dt = dataset_class('/tmp/', download=True)  # Instantiate
    datasets.add(dt)


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
    result = ListConcurrency()
    for name, dt in selected_datasets.items():
        t = threading.Thread(target=worker, args=(result, dt, name), name=name)
        threads.append(t)
        t.start()

    # Join threads
    for t in threads:
        t.join()
    print('That took {} seconds'.format(time.time() - starttime))

    # Create the dict with all datasets already instantiated
    final_datasets = {dt_name: dataset_instance for dt_name, dataset_instance in zip(names, result.list)}
    return final_datasets

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
    datasets_names = ['CWRU', 'IMS', 'MFPT', 'RPDBCS', 'UOC', 'XJTU', 'SEU']
    # datasets_names = ['MFPT']
    raw_datasets = load_datasets(datasets_names)


    root_dir = Path('/tmp/')
    transforms = None
    convert_datasets(raw_datasets, root_dir)
    deep_datasets = instantiate_deep_datasets(root_dir, transforms, datasets_names)


if __name__ == "__main__":
    main()
