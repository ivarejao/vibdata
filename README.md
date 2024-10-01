# Vibdata

Vibdata is an open-source repository that provides a common interface for developers to manipulate vibrational data . It uses a Python package to store and manipulate data, aiming to facilitate the development of machine learning techniques in the context of intelligent fault diagnosis for rotating machinery.

## Features

- Provides a unified interface for handling vibrational data
- Includes five public datasets: 
    - CWRU - Case Western Reserve University Bearing Dataset;
    - IMS - Intelligent Maintenance System Bearing Dataset;
    - UOC - University of Connecticut Gearbox Dataset;
    - PU - Padeborn University Bearing Dataset ;
    - MFPT - Machinery Failure Prevention Technology Bearing Dataset.
- Divided into two main modules:
  - `raw`: Structures datasets into classes for easy and practical data manipulation
  - `deep`: Offers a lazy implementation compatible with deep learning frameworks, especially PyTorch

## Installation
Installing Vibdata is quick and easy! Just follow these simple steps:

1. Ensure you have Python installed on your system*.
2. Install Vibdata using pip:
```bash
pip install git+https://github.com/ivarejao/vibdata.git
```
That's it! You're now ready to start using Vibdata in your projects.

> Note: Vibdata requires Python version 3.10 or later, but is not yet compatible with Python 3.12. Please ensure your Python version is within this range for optimal performance.

## Usage

Here's a simple example of how to use Vibdata:

```python
from vibdata.raw import MFPT_raw

root_dir = "MY_DATASET_DIR"  # Where to save and load datasets
D = MFPT_raw(root_dir, download=True)  # Downloads the dataset if not available

print(D.getMetaInfo())  # Prints metainfo of the dataset
print("Number of signals:", len(D))

# Get the temporal signal and the respectively sample rate
signal = D[0]
signal_metainfo = D.getMetaInfo().iloc[0]
print("Length of signal:", len(signal))
print("Signal sample rate:", signal_metainfo["sample_rate"])
```

## Transformations

Vibdata provides common transformations used in machinery signal processing. You can implement new transformations using the `Transformer` interface, which is compatible with scikit-learn pipelines.

Example of using transformations:

```python
from sklearn.pipeline import make_pipeline
from vibdata.raw import MFPT_raw
from vibdata.deep.signal.transforms import Split, FFT, FilterByValue
from vibdata.datahandler.transforms.TransformDataset import transform_and_saveDataset

transforms = make_pipeline(
    FilterByValue(on_field='sample_rate', values=48828)
    Split(1024),
    FFT(),
    asType(np.float32, on_field='signal'),
)

root_dir = "MY_DATASET_DIR"
D = MFPT_raw(root_dir, download=True)
D_transformed = transform_and_saveDataset(D, transforms, root_dir)
print(D_transformed[3]['signal'].shape)
```

## Authors

- Igor Mattos dos Santos Varejão
- Luciano Henrique Silva Peixoto
- Lucas Gabriel de Oliveira Costa
- Joluan Zucateli 
- Lucas Henrique Sousa Mello


## License

Vibdata is released under the MIT License. This means you are free to use, modify, distribute, and sell this software, provided you include the above copyright notice in all copies or substantial portions of the software.