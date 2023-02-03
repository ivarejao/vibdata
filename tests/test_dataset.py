from termcolor import colored, cprint
import vibdata.datahandler as datahandler
from vibdata.datahandler.base import RawVibrationDataset
from argparse import ArgumentParser

def parse_args():
    """
    Parse args
    """
    parser = ArgumentParser()
    parser.add_argument("--root-dir", help="The directory where data will be stored", required=True)

    args = parser.parse_args()
    return args

def test_dataset(dataset_name: str, dataset_class : RawVibrationDataset) -> None:
    """
    Test the essencial methods of RawVibrationDataset
    """
    space_len = (13-len(dataset_name))//2
    print(colored("---------------\n" +
                  "|" + " "*space_len + dataset_name + " "*space_len + "|\n" +
                  "---------------\n", color="green", attrs=['bold']), end='\n\n')

    meta = dataset_class.getMetaInfo()
    cprint("Labels:", color="red")
    [print(l)for l in dataset_class.getLabelsNames()]
    print("\nMetainfo.shape: ", meta.shape)

    n_samples = meta.shape[0]

    print(f"META:\n {meta.head()}\n\n")

    print(colored("ONE ITEM:\n", color='yellow'))
    sample = dataset_class[0]
    print(f"RAW:\n {sample['signal']}", end="\n\n")
    print(f"META:\n {sample['metainfo']}")

    print(colored("\nSLICE ITEM:\n", color='blue'))
    sample = dataset_class[n_samples-8:n_samples:2]
    print(f"RAW:\n {sample['signal']}", end="\n\n")
    print(f"META:\n {sample['metainfo']}")

    # Test each label
    phrase = " SINGLE CLASS "
    space_len = (13 - len(phrase)) // 2
    print(colored("================\n" +
                  "||" + " "*space_len + phrase + " "*space_len + "||\n" +
                  "================\n", color="green", attrs=['bold']), end='\n\n')

    for l in dataset_class.getLabelsNames():
        print(f"LABEL: {l}")
        print(meta.loc[meta['label'] == l].iloc[0], end='\n--------------------------\n')


if __name__ == "__main__":
    # Define the arguments
    args = parse_args()
    root_dir = args.root_dir

    # modules = [datahandler.CWRU_raw, datahandler.EAS_raw, datahandler.IMS_raw, datahandler.MAFAULDA_raw, datahandler.MFPT_raw, datahandler.PU_raw, datahandler.SEU_raw,
    #            datahandler.UOC_raw, datahandler.XJTU_raw]
    modules = [datahandler.CWRU_raw]
    for dataset in modules:
        D = dataset(root_dir, download=True)
        test_dataset(D.name(), D)

