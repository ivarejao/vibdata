# Vibenet

This is a package made to help deep learning approach into the problem of bearing fault classification
using a variety of open bearing datasets.

![image](./images/Datasets.png "Datasets")

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

### Common transformations
```python
from vibdata.datahandler.transforms.signal import FilterByValue, StandardScaler, MinMaxScaler, Split, FFT, asType, SelectFields, toBinaryClassification, NormalizeSampleRate, Sampling
import pandas as pd

COMMON_TRANSFORMERS = [
    # sampling for testing purporses only. Use whole dataset when everything is working properly.
    Sampling(0.5),
    NormalizeSampleRate(97656),
    Split(6101*2),
    FFT(discard_first_points=1),  # run fft and discard the first point of each signal.
    StandardScaler(on_field='signal', type='all'),
    asType(np.float32, on_field='signal'),
    # toBinaryClassification(), # converts the problem to binary classification
    SelectFields('signal', ['label', 'index'])]


CWRU_TRANSFORMERS = COMMON_TRANSFORMERS

MFPT_TRANSFORMERS = [
    FilterByValue(on_field='dir_name', values=["1 - Three Baseline Conditions",
                                               "3 - Seven More Outer Race Fault Conditions",
                                               "4 - Seven Inner Race Fault Conditions"]),
] + COMMON_TRANSFORMERS

SEU_TRANSFORMERS = [FilterByValue(on_field='channel', values=1)] + COMMON_TRANSFORMERS

PU_TRANSFORMERS = COMMON_TRANSFORMERS

RPDBCS_TRANSFORMERS = [StandardScaler(on_field='signal', type='all'),
                       SelectFields('signal', ['label', 'index'])]


def custom_xjtu_filter(data: dict):
    metainfo: pd.DataFrame = data['metainfo']
    mask = metainfo['fault'].notna()
    metainfo = metainfo[mask].copy()
    signal = data['signal'][mask]
    signal = np.hstack(signal).T
    metainfo['label'] = pd.factorize(metainfo['fault'])[0]
    metainfo = pd.DataFrame(metainfo.values.repeat(2, axis=0),
                            columns=metainfo.columns)

    return {'signal': signal,
            'metainfo': metainfo}


XJTU_TRANSFORMERS = [
    FilterByValue(on_field='intensity', values=[0, 100]),
    custom_xjtu_filter,
] + COMMON_TRANSFORMERS
```

## Sample classification
```python
from vibdata.datahandler import MFPT_raw
from vibdata.datahandler.transforms.TransformDataset import transform_and_saveDataset
from skorch.dataset import ValidSplit # pip install skorch
from skorch.classifier import NeuralNetClassifier
from pytorch_balanced_sampler.sampler import BalancedDataLoader # pip install git+https://github.com/Lucashsmello/pytorch-balanced-sampler
from adabelief_pytorch import AdaBelief # pip install adabelief-pytorch
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn
from torch.utils.data.dataset import Subset
from sklearn.metrics import f1_score
import numpy as np


####Class and functions for transforming data#####

def _transform_output(data):
    """
    Renames data. It just converts the input format into a format that pytorch accepts.
    """
    label = data['label']
    del data['label']
    data['X'] = data['signal']
    del data['signal']
    return data, label


class TransformsDataset(torch.utils.data.IterableDataset):
    def __init__(self, D: torch.utils.data.IterableDataset, transforms) -> None:
        super().__init__()
        self.D = D
        self.transforms = transforms

    def __iter__(self):
        for d in self.D:
            if(hasattr(self.transforms, 'transform')):
                yield self.transforms.transform(d)
            else:
                yield self.transforms(d)

    def __getitem__(self, i):
        d = self.D[i]
        if(hasattr(self.transforms, 'transform')):
            return self.transforms.transform(d)
        else:
            return self.transforms(d)

    def __len__(self):
        return len(self.D)
#########


### Defining dataset ###
D = XJTU_raw('/tmp/', download=True)
cache_dir = '/tmp/cache_signals_data'
D = transform_and_saveDataset(D, XJTU_TRANSFORMERS, cache_dir, batch_size=3000)

Y = D.metainfo['label'].astype(int)
group_ids = D.metainfo['index']
D = TransformsDataset(D, _transform_output)
D.labels = Y
########################

### Defining model and parameters ###


class RPDBCS2020Net(nn.Module):
    def __init__(self, input_size=6100, output_size=8, activation_function=nn.PReLU(), random_state=None):
        if(random_state is not None):
            torch.manual_seed(random_state)
            torch.cuda.manual_seed(random_state)
        super().__init__()
        self.convnet = nn.Sequential(  # input is (n_batches, 1, 6100)
            nn.Conv1d(1, 16, 5, padding=2), activation_function,  # 6100 -> 6096
            nn.MaxPool1d(4, stride=4),  # 6096 -> 1524
            nn.Conv1d(16, 32, 5, padding=2), activation_function,  # 1524 -> 1520
            nn.MaxPool1d(4, stride=4),  # 1520 -> 380
            nn.Conv1d(32, 64, 5, padding=2), activation_function,  # 380 -> 376
            nn.MaxPool1d(4, stride=4),  # 376 -> 94
            nn.Flatten(),
        )

        n = input_size//4//4//4
        self.fc = nn.Sequential(nn.Dropout(0.1), nn.Linear(64 * n, 192),
                                activation_function,
                                nn.Linear(192, output_size)
                                )

        # print('>>>Number of parameters: ',sum(p.numel() for p in self.parameters()))

    def forward(self, X, **kwargs):
        if(X.dim() == 2):
            X = X.reshape(X.shape[0], 1, X.shape[1])
        output = self.convnet(X)
        output = self.fc(output)
        return output


DEFAULT_NET_PARAMS = {
    'device': 'cuda',
    'criterion': nn.CrossEntropyLoss,
    'max_epochs': 100,
    'batch_size': 128,
    'train_split': ValidSplit(0.1, stratified=True),
    'iterator_train': BalancedDataLoader, 'iterator_train__shuffle': True
}

DEFAULT_OPTIM_PARAMS = {'weight_decay': 1e-4, 'lr': 1e-3,
                        'eps': 1e-16, 'betas': (0.9, 0.999),
                        'weight_decouple': False, 'rectify': False,
                        'print_change_log': False}
DEFAULT_OPTIM_PARAMS = {"optimizer__"+key: v for key, v in DEFAULT_OPTIM_PARAMS.items()}
DEFAULT_OPTIM_PARAMS['optimizer'] = AdaBelief

module_params = {
    'module__output_size': len(np.unique(Y)),
    'module__input_size': 6100
}

net = NeuralNetClassifier(module=RPDBCS2020Net, **module_params, **DEFAULT_NET_PARAMS, **DEFAULT_OPTIM_PARAMS)

#####################################

### Evaluation ###
sampler = StratifiedGroupKFold(n_splits=4, shuffle=True)
train_idxs, test_idxs = next(sampler.split(Y, Y, group_ids))
Ytrain, Ytest = Y[train_idxs], Y[test_idxs]
Dtrain = Subset(D, train_idxs)
Dtest = Subset(D, test_idxs)

net.fit(Dtrain, Ytrain)
Ypred = net.predict(Dtest)
score = f1_score(Ytest, Ypred, average='macro')
print("F1-macro:", score)
##################

```

# Test code
Run `python -m pytest`. Requires installing `pytest`.

# Contact

**Author**: Lucas Henrique Sousa Mello, Igor Varejão, Joluan

**Author email**: lucashsmello@gmail.com
