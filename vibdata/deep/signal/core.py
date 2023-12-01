import numpy as np
import pandas as pd
from typing import TypedDict

class SignalSample(TypedDict):
    """
    (signal) : is a 2d np.array that store one or more signals
    (metainfo) : if is an individual sample, metainfo is a pd.Series, otherwise, its a pd.Dataframe
    """
    signal : np.array
    metainfo : pd.DataFrame | pd.Series