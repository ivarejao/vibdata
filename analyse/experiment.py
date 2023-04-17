# Code made in Pycharm by Igor Varejao
import os
import threading
import time
from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
from torch.utils.data import DataLoader
from vibdata.deep.DeepDataset import convertDataset, DeepDataset
import vibdata.raw as datahandler
from vibdata.raw.base import RawVibrationDataset
from vibdata.deep.signal.transforms import Sequential, SplitSampleRate, Split, NormalizeSampleRate
from tqdm import tqdm
from functools import reduce
import logging
from argparse import ArgumentParser

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--datasets', help="Define the dataset which will run, if no one is specified all datasets will run.")
    parser.add_argument('--exp-name', help="Define the exp name")
    
    args = parser.parse_args()

    return args

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
    dt = dataset_class('/tmp/datasets/', download=True)  # Instantiate
    datasets.add(name, dt)


def load_datasets(names: List[str], root_dir : Path) -> Dict[str, RawVibrationDataset]:
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
#    threads = []
#    result = DictConcurrency()
#    for name, dt in selected_datasets.items():
#        t = threading.Thread(target=worker, args=(result, dt, name), name=name)
#        threads.append(t)
#        t.start()
#
#    # Join threads
#    for t in threads:
#        t.join()
    datasets = {}
    for name, class_dt in selected_datasets.items():
        datasets[name] = class_dt(root_dir, download=True)
    print('That took {} seconds'.format(time.time() - starttime))

    return datasets

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
    args = parse_args()
    # Create log config
    log_name = args.exp_name + ".txt" if args.exp_name else f"log-{time.time()}.txt"
    logging.basicConfig(filename=log_name, encoding='utf-8', level=logging.DEBUG)
    # Datasets used
    if not args.datasets:
        datasets_names = ['CWRU', 'IMS', 'MAFAULDA', 'MFPT', 'UOC', 'XJTU']
    else:
        datasets_names = args.datasets.split(",")
    print(datasets_names)
    # datasets_names = ['MAFAULDA']
    root_dir = Path('/home/ivarejao/datasets/')
    # Load the raw datasets and download if it needed
    raw_datasets = load_datasets(datasets_names, root_dir)

    logging.debug("Original size")
    for name, dt in raw_datasets.items():
        logging.debug(f"[{dt.name()}] {len(dt)}")
    logging.debug("\n")

    # Convert the datasets to deep datasets
    biggest_sample_rate = 97656  # MFPT
    transforms = None
    #transforms = Sequential([SplitSampleRate(), NormalizeSampleRate(biggest_sample_rate)])
    #transforms = Sequential([SplitSampleRate()])
    convert_datasets(raw_datasets, root_dir)

    # Insantiate the datasets
    deep_datasets = instantiate_deep_datasets(root_dir, transforms, datasets_names)
    
    # Begin to traverse over the datasets, the tranformation occurs when the data is acessed
    # Storage data
    overall_sizes = set()
    samples = {'dataset' : [], 'size' : [], 'sample_rate': []}
    print("No transform applied")
    # Begin traverse
    for name, dataset in zip(datasets_names, deep_datasets):
        loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=lambda x:x)
        total = 0
        local_sizes = set()
        with tqdm(total=len(loader), desc="Running: ") as pb:
            for ret in loader:
                sizeof = lambda x : 1 if isinstance(x['metainfo'], pd.Series) else x['signal'].shape[0]
                total += reduce(lambda x,y: x + y, [sizeof(r) for r in ret])
                for r in ret:
                    if  (isinstance(r['metainfo'], pd.Series) and r['signal'].shape[0] != 1) or \
                        (isinstance(r['metainfo'], pd.DataFrame) and (r['signal'].shape[0] != r['metainfo'].shape[0])):
                        print("Mismatch size!!!")
                        breakpoint()
                # Add sample_size
                for sig in ret:
                    local_sizes.add(sig['signal'].shape[1])
                    if isinstance(sig['metainfo'], pd.Series):
                        samples['sample_rate'].append(sig['metainfo']['sample_rate'])
                        samples['size'].append(sig['signal'].shape[1])
                        samples['dataset'].append(name)
                    else:
                        for i in range(sig['metainfo'].shape[0]):
                            samples['size'].append(sig['signal'][i].size)
                            samples['sample_rate'].append(sig['metainfo'].iloc[i]['sample_rate'])
                            samples['dataset'].append(name)
                pb.update()
        logging.info(f"[{name}] Total samples : {total}")
        logging.info(f"Num: {local_sizes}")
        overall_sizes = overall_sizes | local_sizes
    logging.info(f"Overall sizes: {overall_sizes}")
    
    samples_df = pd.DataFrame(samples).to_csv('samples_per_datasets_transformed.csv', index=False)

if __name__ == "__main__":
    main()
