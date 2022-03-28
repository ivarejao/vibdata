# Code made in Pycharm by Igor Varejao
from termcolor import colored, cprint

from vibdata.datahandler.IMS.IMS import IMS_raw
from vibdata.datahandler.UOC.UOC import UOC_raw
from vibdata.datahandler.XJTU.XJTU import XJTU_raw
from vibdata.datahandler.MFPT.MFPT import MFPT_raw
from vibdata.datahandler.SEU.SEU import SEU_raw
from vibdata.datahandler.MAFAULDA.MAFAULDA import MAFAULDA_raw

def testDataset(dataset_name: str, dataset_class):
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
    D = MAFAULDA_raw('/tmp', download=True)
    testDataset("MAFAULDA", D)



