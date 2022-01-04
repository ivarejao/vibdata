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

    def asSimpleForm(self):
        dataset = self._getDataset()
        metainfo = self.getMetaInfo()
        n = len(dataset)
        signal = np.empty(shape=(n, 1024*100), dtype=np.float32)
        for i in tqdm(range(n), desc="Loading RPDBCS dataset"):
            signal[i] = dataset.getSignal(i).time.getY()[:1024*100]
        return {'signal': signal,
                'metainfo': metainfo}

    def getLabelsNames(self):
        return ['Normal', 'Roçamento', 'Problemas na medição', 'Desalinhamento', 'Desbalanceamento']
