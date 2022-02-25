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

## Changes
Replace the `download_file_from_google_drive` of `py torchvision.datasets.utils` to this code:
```python
def download_file_from_google_drive(file_id: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None):
    """Download a Google Drive file from  and place it in root.

    Args:
        file_id (str): id of file to be downloaded
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the id of the file.
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests
    url = "https://docs.google.com/uc?export=download"

    root = os.path.expanduser(root)
    if not filename:
        filename = file_id
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={'id': file_id}, stream=True)
        token = _get_confirm_token(response)

        if ("Download anyway" in str(response.content)):  # Igor
            response = session.get(url, params={'id': file_id, 'confirm': True}, stream=True)  # Igor

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        # Ideally, one would use response.status_code to check for quota limits, but google drive is not consistent
        # with their own API, refer https://github.com/pytorch/vision/issues/2992#issuecomment-730614517.
        # Should this be fixed at some place in future, one could refactor the following to no longer rely on decoding
        # the first_chunk of the payload
        response_content_generator = response.iter_content(32768)
        first_chunk = None
        while not first_chunk:  # filter out keep-alive new chunks
            first_chunk = next(response_content_generator)

        if _quota_exceeded(first_chunk):
            msg = (
                f"The daily quota of the file {filename} is exceeded and it "
                f"can't be downloaded. This is a limitation of Google Drive "
                f"and can only be overcome by trying again later."
            )
            raise RuntimeError(msg)

        _save_response_content(itertools.chain((first_chunk, ), response_content_generator), fpath)
        response.close()
```
