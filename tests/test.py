from vibdata.datahandler import MFPT_raw
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('root_dir', type=str,  help='Root directory')

args = parser.parse_args()


root_dir = args.root_dir
D = MFPT_raw(root_dir, download=True)
print(D.getMetaInfo())
