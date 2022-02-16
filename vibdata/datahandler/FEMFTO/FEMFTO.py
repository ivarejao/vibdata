# Code made in Pycharm by Igor Varejao
import os
from importlib import resources

import numpy as np
import pandas as pd

from vibdata.datahandler.base import RawVibrationDataset, DownloadableDataset

class FEMFTO_raw(RawVibrationDataset, DownloadableDataset):
