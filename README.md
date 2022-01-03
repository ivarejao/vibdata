# Vibenet

This is a package made to help deep learning approach into the problem of bearing fault classification
using a variety of open bearing datasets.

![image](Datasets.png "Datasets")

# Instructions 

## Download a dataset
You can download automatically a dataset to any directory, if the dataset is not already available locally:
```python
from datahandler import MFPT_raw

root_dir = "MY_DATASET_DIR" # Where to save and load datasets.
D = MFPT_raw(root_dir, download=True) # This will download the dataset to root_dir if not already available in root_dir.
print(D.getMetaInfo()) # prints metainfo of the dataset
print("")
print("Number of signals:", len(D.asSimpleForm()['signal'])) # Prints the number of signals.
```

## Transformations
This package provides some transformations commonly used in machinery signal processing (see [signal.py](datahandler/transforms/signal.py)).
It is possible to implement more new transformations using the interface [Transformer](datahandler/transforms/signal.py#L15). It implements the transform method and inherits sklearn BaseEstimator, so it can be directly used in a sklearn pipeline: 

```python
from sklearn.pipeline import make_pipeline
from datahandler import MFPT_raw
from datahandler.transforms.signal import *

transforms = make_pipeline( StandardScaler(on_field='signal', type='all'),
                            Split(1024),
                            FFT(),
                            asType(np.float32, on_field='signal'),
                            toBinaryClassification(),
                            SelectFields('signal', 'label'))


root_dir = "MY_DATASET_DIR"
D = MFPT_raw(root_dir, download=True)
D_transformed = transforms.transform(D.asSimpleForm())
print(D_transformed['signal'].shape)
```

# Test code
Run `python -m pytest`. Requires installing `pytest`.

# Contact

**Author**: Lucas Henrique Sousa Mello

**Author email**: lucashsmello@gmail.com
