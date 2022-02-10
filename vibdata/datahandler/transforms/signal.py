from abc import abstractmethod
from typing import Dict, List
from sklearn import preprocessing
from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd
from scipy import interpolate


class Transform(BaseEstimator):
    @abstractmethod
    def transform(self, data):
        pass

    def fit(self, *args, **kwargs):
        return self

    def __call__(self, data):
        return self.transform(data)


class Sampling(Transform):
    def __init__(self, ratio: float, random_state=None) -> None:
        super().__init__()
        self.random_state = random_state
        self.ratio = ratio
        self.R = np.random.RandomState(self.random_state)

    def transform(self, data):
        sigs = data['signal']
        metainfo = data['metainfo']
        n = len(sigs)
        idxs = self.R.choice(n, int(self.ratio*n), replace=False)
        if(isinstance(sigs, list)):
            sigs = [sigs[i] for i in idxs]
        else:
            sigs = sigs[idxs]
        metainfo = metainfo.iloc[idxs]
        return {'signal': sigs, 'metainfo': metainfo}


class TransformOnField(Transform):
    def __init__(self, transformer: Transform, on_field=None) -> None:
        super().__init__()
        self.on_field = on_field
        self.transformer = transformer

    def transform(self, data):
        if(self.on_field is None):
            return self.transformer.transform(data)
        if(isinstance(data, dict)):
            data = data.copy()
        else:
            data = data.copy(deep=False)
        data[self.on_field] = self.transformer.transform(data[self.on_field])
        return data


class TransformOnFieldClass(Transform):
    def __init__(self, on_field=None) -> None:
        self.on_field = on_field

    @abstractmethod
    def transform_(self, data):
        pass

    def transform(self, data):
        if(self.on_field is None):
            return self.transform_(data)
        if(isinstance(data, dict)):
            data = data.copy()
        else:
            data = data.copy(deep=False)
        data[self.on_field] = self.transform_(data[self.on_field])
        return data


class FilterByValue(Transform):
    def __init__(self, on_field, values, remove=False) -> None:
        super().__init__()
        self.on_field = on_field
        if(isinstance(values, str)):
            self.values = [values]
        elif(hasattr(values, '__iter__') or hasattr(values, '__getitem__')):
            self.values = values
        else:
            self.values = [values]
        self.remove = remove

    def transform(self, data):
        D = data['metainfo'][self.on_field]
        valid_values = D.isin(self.values)
        if(self.remove):
            valid_values = ~valid_values
        data = data.copy()
        data['metainfo'] = data['metainfo'][valid_values]
        data['signal'] = np.array(data['signal'], dtype=object)[valid_values]
        return data


class toNumpy(TransformOnFieldClass):
    def __init__(self, on_field=None) -> None:
        super().__init__(on_field=on_field)

    def transform_(self, data):
        if(isinstance(data, (pd.DataFrame, pd.core.series.Series))):
            return data.values
        return data


class asType(TransformOnFieldClass):
    def __init__(self, dtype, on_field=None) -> None:
        super().__init__(on_field=on_field)
        self.dtype = dtype

    def transform_(self, data):
        if(not isinstance(data, np.ndarray)):
            return np.array(data, dtype=self.dtype)
        return data.astype(self.dtype)


class Sequential(Transform):
    def __init__(self, *transforms):
        super().__init__()
        self.transforms: List[Transform] = list(transforms)

    def transform(self, data):
        for t in self.transforms:
            if(t is None):
                continue
            if(hasattr(t, 'transform')):
                data = t.transform(data)
            else:
                data = t(data)
        return data

    def append(self, other: Transform) -> None:
        self.transforms.append(other)


class Split(Transform):
    """
    Not to be used at a Pipeline with an classifier/regressor.
    In place.
    """

    def __init__(self, window_size, on_field='signal') -> None:
        self.window_size = window_size
        self.on_field = on_field

    def transform(self, data: Dict):
        data = data.copy()
        sigs = data[self.on_field]
        metainfo = data['metainfo'].copy(deep=False)
        ret = []
        for s in sigs:
            if(len(s) < self.window_size):
                snew = np.zeros(self.window_size, dtype=s.dtype)
                snew[:len(s)] = s
                s = snew
            k = len(s) % self.window_size
            if(k > 0):
                s = s[:-k]
            assert(len(s) > 2)
            s = s.reshape(-1, self.window_size)
            ret.append(s)

        metainfo[self.on_field] = ret
        metainfo = metainfo.explode(self.on_field)

        data['metainfo'] = metainfo.drop(self.on_field, axis=1)
        data[self.on_field] = np.stack(metainfo[self.on_field].values)

        return data


class FFT(TransformOnFieldClass):
    def __init__(self, on_field='signal', discard_first_points=0) -> None:
        super().__init__(on_field=on_field)
        self.discard_first_points = discard_first_points

    def transform_(self, X):
        ret = []
        for x in X:
            x = np.fft.fft(x)
            x = 2 * np.abs(x) / len(x)
            n = x.shape[0]
            x = x[self.discard_first_points:n//2]
            ret.append(x)
        return ret


class SelectFields(Transform):
    def __init__(self, fields, metainfo_fields=[]):
        super().__init__()
        self.fields = [fields] if isinstance(fields, str) else fields
        self.metainfo_fields = [metainfo_fields] if isinstance(metainfo_fields, str) else metainfo_fields

    def transform(self, data):
        ret = {f: data[f] for f in self.fields}
        metainfo = data['metainfo']
        ret.update({f: metainfo[f].values if f != 'index' else metainfo.index.values for f in self.metainfo_fields})
        return ret


class ReshapeSingleChannel(TransformOnFieldClass):
    def __init__(self, on_field='signal') -> None:
        super().__init__(on_field=on_field)

    def transform_(self, data):
        return data.reshape(len(data), 1, -1)


class SklearnFitTransform(TransformOnFieldClass):
    def __init__(self, transformer: TransformerMixin, on_field=None, on_row=False):
        super().__init__(on_field=on_field)
        self.transformer = transformer
        self.on_row = on_row

    def transform_(self, data):
        if(self.on_row):
            return self.transformer.fit_transform(data.T).T
        return self.transformer.fit_transform(data)


class NormalizeSampleRate(Transform):
    def __init__(self, sample_rate) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    def transform(self, data):
        data = data.copy()
        metainfo = data['metainfo'].copy(deep=True)
        sr = metainfo['sample_rate']
        sigs = data['signal']
        new_sigs_list = []
        for i, sig in enumerate(sigs):
            n = len(sig)
            ratio = self.sample_rate/sr.iloc[i]
            X = np.arange(len(sig))
            f_interpolate = interpolate.interp1d(X, sig, kind='linear')
            Xt = np.linspace(0, n-1, int(np.round(ratio*(n-1)+1)))
            new_sig = f_interpolate(Xt)
            new_sigs_list.append(new_sig)
        metainfo['sample_rate'] = self.sample_rate
        metainfo['old_sample_rate'] = sr
        data['signal'] = new_sigs_list

        return data


class StandardScaler(TransformOnFieldClass):
    def __init__(self, on_field=None, type='all') -> None:
        super().__init__(on_field=on_field)
        self.type = type

    def transform_(self, data):
        ret = []
        if(self.type == 'row'):
            for row in data:
                ret.append((row-row.mean())/row.std())
        elif(self.type == 'all'):
            data_flatten = np.hstack(data)
            m, s = data_flatten.mean(), data_flatten.std()
            ret = [(row-m)/s for row in data]
        else:
            raise NotImplemented
        return ret


class MinMaxScaler(TransformOnFieldClass):
    def __init__(self, on_field=None, type='all') -> None:
        super().__init__(on_field=on_field)
        self.type = type

    def transform_(self, data):
        ret = []
        if(self.type == 'row'):
            for row in data:
                r = row-row.min()
                ret.append(r/r.max())
        elif(self.type == 'all'):
            data_flatten = np.hstack(data)
            mn, mx = data_flatten.min(), data_flatten.max()
            ret = [(row-mn)/(mx-mn) for row in data]
        else:
            raise NotImplemented
        return ret


class toBinaryClassification(Transform):
    def __init__(self, negative_label=0) -> None:
        super().__init__()
        self.negative_label = negative_label

    def transform(self, data):
        data = data.copy()
        metainfo = data['metainfo'].copy(deep=False)
        mask = metainfo['label'] == self.negative_label
        metainfo.loc[mask, 'label'] = 0
        metainfo.loc[~mask, 'label'] = 1
        data['metainfo'] = metainfo
        return data
