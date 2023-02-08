import argparse
import random

from termcolor import colored, cprint
import vibdata.datahandler as datahandler
from vibdata.datahandler.base import RawVibrationDataset
from argparse import ArgumentParser
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args() -> argparse.Namespace:
    """
    Parse args

    Returns:
        Argument parsed into kind of dict
    """
    parser = ArgumentParser()
    parser.add_argument("--root-dir", help="The directory where data will be stored", required=True)

    args = parser.parse_args()
    return args

def test_dataset(dataset_name: str, dataset_class : RawVibrationDataset) -> None:
    """
    Test the essential methods of RawVibrationDataset including:
    - If the metainfo is cohesive, as the number of multi_samples
    - If the array by the __getitem__  method is working properly

    Args:
        dataset_name (str): The dataset name that is being tested
        dataset_class (RawVibrationDataset): The dataset class
    Returns:
        None
    """
    cprint(f"< {dataset_name} >", color="green", attrs=['bold'], end='\n\n')

    # Imprime os nomes dos labels
    cprint("Labels:", color="red")
    meta = dataset_class.getMetaInfo()
    labels_name =  dataset_class.getLabels(as_str=True)
    labels_id = dataset_class.getLabels()
    for i, label in zip(labels_id, labels_name):
        print(str(i) + ". " + label)
    # Print some general info
    n_samples, n_features = meta.shape
    print(f"Signals amount: {n_samples}")
    print(f"Number of metainfo features: {n_features}")
    print()

    # Test one item
    print(colored("< One item >", color='yellow'))
    # Pick an index randomly
    idx = random.randint(0, n_samples-1)
    sample = dataset_class[idx]
    # Check the __getitem__ from an integer
    assert len(sample['signal']) == sample['metainfo'].shape[0]
    assert len(sample['signal']) == 1
    print("Raw(%d):" % len(sample['signal']))
    print(f"{sample['signal']}", end="\n\n")
    print(f"Meta({sample['metainfo'].iloc[0].shape}):")
    print(f"{sample['metainfo']}")
    print()

    # Check the __getitem__  from a slice
    cprint("< Slice item >", color='blue')
    idx = random.randint(0, n_samples-1)
    multi_samples = dataset_class[(idx-8):idx:2]
    assert len(multi_samples['signal']) == multi_samples['metainfo'].shape[0]
    assert len(multi_samples['signal']) == 4
    print("Raw(%d):" % len(multi_samples['signal']))
    print(f"{sample['signal']}", end="\n\n")
    print(f"Meta({sample['metainfo'].shape}):")
    print(f"{sample['metainfo']}")
    print()

    # Test each label
    cprint(f"< single class >", color="green", attrs=['bold'])
    sns.set_theme()
    for l_id, label in zip(labels_id, labels_name):
        cprint(f"Label: {label}", color="green")
        idxs = meta.loc[meta['label'] == l_id].index.values
        final_sample = idxs[random.randint(0, len(idxs)-1)]

        # Plot the signal
        fig = plt.figure(figsize=(14, 7))
        sns.lineplot(*dataset_class[final_sample]['signal'], color='black')
        plt.title(f"{dataset_class.name()}: {label} signal")
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.show(block=True)

if __name__ == "__main__":
    # Define the arguments
    args = parse_args()
    root_dir = args.root_dir

    modules = [datahandler.CWRU_raw]
    for dataset in modules:
        D = dataset(root_dir, download=True)
        test_dataset(D.name(), D)

