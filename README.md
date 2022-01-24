# Vibenet

This is a package made to help deep learning approach into the problem of bearing fault classification
using a variety of open bearing datasets.

![image](Datasets.png "Datasets")

# Instructions 

## Download a dataset
You can download automatically a dataset to any directory, if the dataset is not already available locally:
```python
from vibdata.datahandler import MFPT_raw

root_dir = "MY_DATASET_DIR" # Where to save and load datasets.
D = MFPT_raw(root_dir, download=True) # This will download the dataset to root_dir if not already available in root_dir.
print(D.getMetaInfo()) # prints metainfo of the dataset
print("")
print("Number of signals:", len(D)) # Prints the number of signals.
```

## Transformations
This package provides some transformations commonly used in machinery signal processing (see [signal.py](vibdata/datahandler/transforms/signal.py)).
It is possible to implement more new transformations using the interface [Transformer](vibdata/datahandler/transforms/signal.py#L14). It implements the transform method and inherits sklearn BaseEstimator, so it can be directly used in a sklearn pipeline: 

```python
from sklearn.pipeline import make_pipeline
from vibdata.datahandler import MFPT_raw
from vibdata.datahandler.transforms.signal import *
from vibdata.datahandler.transforms.TransformDataset import transform_and_saveDataset

transforms = make_pipeline( Split(1024),
                            FFT(),
                            StandardScaler(on_field='signal', type='all'),
                            asType(np.float32, on_field='signal'),
                            toBinaryClassification(),
                            SelectFields('signal', 'label'))


root_dir = "MY_DATASET_DIR"
D = MFPT_raw(root_dir, download=True)
D_transformed = transform_and_saveDataset(D, transforms, root_dir)
print(D_transformed[3]['signal'].shape)
```

# Test code
Run `python -m pytest`. Requires installing `pytest`.

# Contact

**Author**: Lucas Henrique Sousa Mello, Igor Varejão, Joluan

**Author email**: lucashsmello@gmail.com
