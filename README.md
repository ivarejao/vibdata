# Vibenet

This is a package made to help deep learning approach into the problem of bearing fault classification
using a variety of open bearing datasets.

![image](Datasets.png "Datasets")

# Instructions 

## Download a dataset
```python
from datahandler import MFPT_raw

root_dir = "MY_DATASET_DIR" # Where to save and load datasets.
D = MFPT_raw(root_dir, download=True) # This will download the dataset to root_dir if not already available in root_dir.
print(D.getMetaInfo()) # prints metainfo of the dataset
print("")
print("Number of signals:", len(D.asSimpleForm()['signal'])) # Prints the number of signals.
```


# Contact

**Author**: Lucas Henrique Sousa Mello

**Author email**: lucashsmello@gmail.com
