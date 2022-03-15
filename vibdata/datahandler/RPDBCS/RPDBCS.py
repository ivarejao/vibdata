from typing import Dict
from ..base import RawVibrationDataset
import pandas as pd
import numpy as np


class RPDBCS_raw(RawVibrationDataset):
    def __init__(self, root_dir: str, frequency_domain=False, **kwargs):
        self.root_dir = root_dir
        self.dataset = None
        self.frequency_domain = frequency_domain

    def _getDataset(self):
        if(self.dataset is None):
            from rpdbcs.datahandler.dataset import readDataset
            if(self.frequency_domain):
                self.dataset = readDataset(feats_file="{}/RPDBCS_raw/labels.csv".format(self.root_dir),
                                           freq_file="{}/RPDBCS_raw/freq.csv".format(self.root_dir),
                                           use_cache=False, npoints=11000, remove_first=100,
                                           dtype=np.float32)
                self.dataset.normalize(37.28941975)
                self.n_points = 6100
            else:
                self.n_points = 1024*100
                self.dataset = readDataset(feats_file="{}/RPDBCS_raw/labels.csv".format(self.root_dir),
                                           time_file="{}/RPDBCS_raw/time".format(self.root_dir), use_cache=False, npoints=self.n_points,
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
            if(self.frequency_domain):
                sig = dataset.getSignal(i).freq.getY()[:self.n_points]
            else:
                sig = dataset.getSignal(i).time.getY()[:self.n_points]
            return {'signal': sig, 'metainfo': df}
        i = np.arange(len(dataset))[i]
        n = len(i)
        if(self.frequency_domain):
            sigs = dataset.asMatrix()[i, :self.n_points]
        else:
            sigs = np.empty((n, self.n_points), dtype=np.float32)
            for k, j in enumerate(i):
                sigs[k] = dataset.getSignal(j).time.getY()[:self.n_points]

        return {'signal': sigs, 'metainfo': df}

    def asSimpleForm(self):
        return self[:]

    def getLabelsNames(self):
        return ['Normal', 'Roçamento', 'Problemas na medição', 'Desalinhamento', 'Desbalanceamento']

