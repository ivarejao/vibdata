from typing import Dict
from ..base import RawVibrationDataset
import pandas as pd
import numpy as np
from tqdm import tqdm


class RPDBCS_raw(RawVibrationDataset):
    def __init__(self, root_dir: str, **kwargs):
        self.root_dir = root_dir
        self.dataset = None

    def _getDataset(self):
        if(self.dataset is None):
            from rpdbcs.datahandler.dataset import readDataset
            self.dataset = readDataset(feats_file="{}/RPDBCS_raw/labels.csv".format(self.root_dir),
                                       time_file="{}/RPDBCS_raw/time".format(self.root_dir), use_cache=False, npoints=1024*100,
                                       dtype=np.float32)
        return self.dataset

    def getMetaInfo(self, labels_as_str=False) -> pd.DataFrame:
        dataset = self._getDataset()
        df = dataset.asDataFrame().copy(False)
        targets, targets_names = dataset.getMulticlassTargets()
        df['label'] = targets
        return df

    def __iter__(self):
        dataset = self._getDataset()
        for i in range(len(dataset)):
            yield self[i]

    def __getitem__(self, i) -> Dict:
        dataset = self._getDataset()
        df = dataset.asDataFrame().iloc[i].copy(False)
        df['label'] = dataset.getMulticlassTargets()[0][i]

        if(isinstance(i, int)):
            return {'signal': dataset.getSignal(i).time.getY()[:1024*100], 'metainfo': df}
        i = np.arange(len(dataset))[i]
        n = len(i)
        sigs = np.empty((n, 1024*100), dtype=np.float32)
        for k, j in enumerate(i):
            sigs[k] = dataset.getSignal(j).time.getY()[:1024*100]

        return {'signal': sigs, 'metainfo': df}

    # def __getitem__(self, i) -> np.ndarray:
    #     dataset = self._getDataset()
    #     df = dataset.asDataFrame().iloc[i].copy(False)
    #     df['label'] = dataset.getMulticlassTargets()[0][i]
    #     df = df.to_dict()

    #     if(isinstance(i, int)):
    #         df['signal'] = dataset.getSignal(i).time.getY()[:1024*100]
    #         return df
    #     i = range(len(dataset))[i]
    #     n = len(i)
    #     sigs = np.empty((n, 1024*100), dtype=np.float32)
    #     for j in i:
    #         sigs[j] = dataset.getSignal(j).time.getY()[:1024*100]

    #     df['signal'] = sigs

    #     return df

    def asSimpleForm(self):
        dataset = self._getDataset()
        metainfo = self.getMetaInfo()
        n = len(dataset)
        signal = np.empty(shape=(n, 1024*100), dtype=np.float32)
        for i in tqdm(range(n), desc="Loading RPDBCS dataset"):
            signal[i] = dataset.getSignal(i).time.getY()[:1024*100]
        return {'signal': signal, 'metainfo': metainfo}

    def getLabelsNames(self):
        return ['Normal', 'Roçamento', 'Problemas na medição', 'Desalinhamento', 'Desbalanceamento']
