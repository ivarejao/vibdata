from abc import abstractmethod
from typing import Dict, List, Literal, TypedDict

import cv2 as cv
import numpy as np
import pandas as pd
from scipy import interpolate
from sklearn import preprocessing
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram, resample_poly
from sklearn.base import BaseEstimator, TransformerMixin

from vibdata.deep.signal.core import SignalSample


class Transform(BaseEstimator, TransformerMixin):
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
        sigs = data["signal"]
        metainfo = data["metainfo"]
        n = len(sigs)
        idxs = self.R.choice(n, int(self.ratio * n), replace=False)
        if isinstance(sigs, list):
            sigs = [sigs[i] for i in idxs]
        else:
            sigs = sigs[idxs]
        metainfo = metainfo.iloc[idxs]
        return {"signal": sigs, "metainfo": metainfo}


class TransformOnField(Transform):
    def __init__(self, transformer: Transform, on_field=None) -> None:
        super().__init__()
        self.on_field = on_field
        self.transformer = transformer

    def transform(self, data):
        if self.on_field is None:
            return self.transformer.transform(data)
        if isinstance(data, dict):
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
        if self.on_field is None:
            return self.transform_(data)
        if isinstance(data, dict):
            data = data.copy()
        else:
            data = data.copy(deep=False)
        data[self.on_field] = self.transform_(data[self.on_field])
        return data


class FilterByValue(Transform):
    def __init__(self, on_field, values, remove=False) -> None:
        super().__init__()
        self.on_field = on_field
        if isinstance(values, str):
            self.values = [values]
        elif hasattr(values, "__iter__") or hasattr(values, "__getitem__"):
            self.values = values
        else:
            self.values = [values]
        self.remove = remove

    def transform(self, data):
        D = data["metainfo"][self.on_field]
        valid_values = D.isin(self.values)
        if self.remove:
            valid_values = ~valid_values
        data = data.copy()
        data["metainfo"] = data["metainfo"][valid_values]
        data["signal"] = np.array(data["signal"], dtype=object)[valid_values]
        return data


class toNumpy(TransformOnFieldClass):
    def __init__(self, on_field=None) -> None:
        super().__init__(on_field=on_field)

    def transform_(self, data):
        if isinstance(data, (pd.DataFrame, pd.core.series.Series)):
            return data.values
        return data


class asType(TransformOnFieldClass):
    def __init__(self, dtype, on_field=None) -> None:
        super().__init__(on_field=on_field)
        self.dtype = dtype

    def transform_(self, data):
        if not isinstance(data, np.ndarray):
            return np.array(data, dtype=self.dtype)
        return data.astype(self.dtype)


class Sequential(Transform):
    def __init__(self, transforms: List[Transform]):
        super().__init__()
        self.transforms = transforms

    def transform(self, data):
        for t in self.transforms:
            if t is None:
                continue
            if hasattr(t, "transform"):
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

    def __init__(self, window_size, on_field="signal") -> None:
        self.window_size = window_size
        self.on_field = on_field

    def transform(self, data: Dict):
        data = data.copy()
        sigs = data[self.on_field]
        metainfo = data["metainfo"].copy(deep=False)
        ret = []
        for s in sigs:
            if len(s) < self.window_size:
                snew = np.zeros(self.window_size, dtype=s.dtype)
                snew[: len(s)] = s
                s = snew
            k = len(s) % self.window_size
            if k > 0:
                s = s[:-k]
            assert len(s) > 2
            s = s.reshape(-1, self.window_size)
            ret.append(s)

        metainfo[self.on_field] = ret
        metainfo = metainfo.explode(self.on_field)

        data["metainfo"] = metainfo.drop(self.on_field, axis=1)
        data[self.on_field] = np.stack(metainfo[self.on_field].values)

        return data


class FFT(Transform):
    def __init__(self, discard_first_points=0) -> None:
        super().__init__()
        self.discard_first_points = discard_first_points

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        signals = data["signal"]

        ret = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            sig_rfft = np.abs(rfft(sig, norm="forward")) * 2  # Amplitudes

            # Low-pass filter
            if "original_sample_rate" in entry:
                sig_sample_rate = entry["sample_rate"]
                sig_freqs = rfftfreq(sig.size, d=1 / sig_sample_rate)

                bandwidth = entry["original_sample_rate"] / 2
                mask = sig_freqs >= bandwidth
                sig_rfft[mask] = 0.0

            ret.append(sig_rfft[self.discard_first_points :])

        data["signal"] = ret
        return data


class Spectrogram(Transform):
    """Spectrogram transform

    See `scipy.signal.spectrogram` for more information about the parameters.

    This transformation adds the `delta_t` and `delta_f` columns in metainfo for time
    and frequency resolutions.
    """

    def __init__(
        self,
        window=("tukey", 0.25),
        nperseg=None,
        noverlap=None,
        nfft=None,
        detrend="constant",
        return_onesided=True,
        scaling="density",
        axis=-1,
        mode="psd",
    ):
        super().__init__()
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.detrend = detrend
        self.return_onesided = return_onesided
        self.scaling = scaling
        self.axis = axis
        self.mode = mode

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=True)
        signals = data["signal"]

        ret = []
        new_metainfo = []
        for (_, entry), sig in zip(metainfo.iterrows(), signals):
            sig_sample_rate = entry["sample_rate"]
            f, t, Sxx = spectrogram(
                sig,
                fs=sig_sample_rate,
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                detrend=self.detrend,
                return_onesided=self.return_onesided,
                scaling=self.scaling,
                axis=self.axis,
                mode=self.mode,
            )

            # Low-pass filter
            if "original_sample_rate" in entry:
                bandwidth = entry["original_sample_rate"] / 2
                mask = f >= bandwidth
                Sxx[mask] = 0.0

            ret.append(Sxx)

            entry["delta_t"] = t[1] - t[0]
            entry["delta_f"] = f[1] - f[0]
            entry["domain"] = "time-frequency"
            new_metainfo.append(entry)

        data["signal"] = ret
        data["metainfo"] = pd.DataFrame(new_metainfo)
        return data


class SelectFields(Transform):
    def __init__(self, fields, metainfo_fields=[]):
        super().__init__()
        self.fields = [fields] if isinstance(fields, str) else fields
        self.metainfo_fields = [metainfo_fields] if isinstance(metainfo_fields, str) else metainfo_fields

    def transform(self, data):
        ret = {f: data[f] for f in self.fields}
        metainfo = data["metainfo"]
        ret.update({f: metainfo[f].values if f != "index" else metainfo.index.values for f in self.metainfo_fields})
        return ret


class ReshapeSingleChannel(TransformOnFieldClass):
    def __init__(self, on_field="signal") -> None:
        super().__init__(on_field=on_field)

    def transform_(self, data):
        return data.reshape(len(data), 1, -1)


class SklearnFitTransform(TransformOnFieldClass):
    def __init__(self, transformer: TransformerMixin, on_field=None, on_row=False):
        super().__init__(on_field=on_field)
        self.transformer = transformer
        self.on_row = on_row

    def transform_(self, data):
        if self.on_row:
            return self.transformer.fit_transform(data.T).T
        return self.transformer.fit_transform(data)


class StandardScaler(TransformOnFieldClass):
    def __init__(self, on_field=None, type="all") -> None:
        super().__init__(on_field=on_field)
        self.type = type

    def transform_(self, data):
        ret = []
        if self.type == "row":
            for row in data:
                ret.append((row - row.mean()) / row.std())
        elif self.type == "all":
            data_flatten = np.hstack(data)
            m, s = data_flatten.mean(), data_flatten.std()
            ret = [(row - m) / s for row in data]
        else:
            raise NotImplemented
        return ret


class MinMaxScaler(TransformOnFieldClass):
    def __init__(self, on_field=None, type="all") -> None:
        super().__init__(on_field=on_field)
        self.type = type

    def transform_(self, data):
        ret = []
        if self.type == "row":
            for row in data:
                r = row - row.min()
                ret.append(r / r.max())
        elif self.type == "all":
            data_flatten = np.hstack(data)
            mn, mx = data_flatten.min(), data_flatten.max()
            ret = [(row - mn) / (mx - mn) for row in data]
        else:
            raise NotImplemented
        return ret


class toBinaryClassification(Transform):
    def __init__(self, negative_label=0) -> None:
        super().__init__()
        self.negative_label = negative_label

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        mask = metainfo["label"] == self.negative_label
        metainfo.loc[mask, "label"] = 0
        metainfo.loc[~mask, "label"] = 1
        data["metainfo"] = metainfo
        return data


#
# Temporal transforms
#


class SplitSampleRate(Transform):
    def __init__(self, on_field="signal") -> None:
        self.on_field = on_field

    def transform(self, data: SignalSample) -> SignalSample:
        sigs = data[self.on_field]
        metainfo = data["metainfo"].copy(deep=False)
        # Trick in order to admit metainfo as a pd.Series or pd.DataFrame
        iter_meta = [(None, metainfo)] if isinstance(metainfo, pd.Series) else metainfo.iterrows()
        # Accumulators variables
        splitted_signals = []
        splitted_metainfo = []

        # Iterate over the signals
        # TODO: Vectorize this process
        for s, (_, sig_metainfo) in zip(sigs, iter_meta):
            # breakpoint()
            window_size = sig_metainfo["sample_rate"]
            if len(s) < window_size:
                snew = np.zeros(window_size, dtype=s.dtype)
                snew[: len(s)] = s
                s = snew
            k = len(s) % window_size
            if k > 0:
                s = s[:-k]
            assert len(s) > 2
            # Do the actual split
            s = s.reshape(-1, window_size)
            # Store the new signals
            splitted_signals.append(s)
            # Clone the metainfo
            splitted_metainfo.append(
                pd.DataFrame(
                    [
                        sig_metainfo,
                    ]
                    * len(s)
                )
            )
        # Just integrate into one object
        splitted_metainfo = pd.concat(splitted_metainfo)
        splitted_signals = np.concatenate(splitted_signals)

        data["metainfo"] = splitted_metainfo
        data[self.on_field] = splitted_signals

        return data


class NormalizeSampleRate(Transform):
    def __init__(self, sample_rate) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        # Trick in order to admit metainfo as a pd.Series or pd.DataFrame
        iter_meta = [(None, metainfo)] if isinstance(metainfo, pd.Series) else metainfo.iterrows()

        sigs = data["signal"]
        new_sigs_list = []
        for sig, (_, metasig) in zip(sigs, iter_meta):
            n = len(sig)
            ratio = self.sample_rate / metasig["sample_rate"]
            X = np.arange(len(sig))
            f_interpolate = interpolate.interp1d(X, sig, kind="linear")
            # Xt = np.linspace(0, n-1, int(np.round(ratio*(n-1)+1)))
            Xt = np.linspace(0, n - 1, self.sample_rate)
            new_sig = f_interpolate(Xt)
            new_sigs_list.append(new_sig)

        metainfo["original_sample_rate"] = metainfo["sample_rate"]
        metainfo["sample_rate"] = self.sample_rate

        # Redefine
        data["metainfo"] = metainfo
        data["signal"] = np.array(new_sigs_list)

        return data


class NormalizeSampleRatePoly(Transform):
    """Normalize sample rate using polyphase filtering"""

    def __init__(self, sample_rate) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    def transform(self, data):
        data = data.copy()
        metainfo = data["metainfo"].copy(deep=False)
        # Trick in order to admit metainfo as a pd.Series or pd.DataFrame
        iter_meta = [(None, metainfo)] if isinstance(metainfo, pd.Series) else metainfo.iterrows()

        sigs = data["signal"]
        new_sigs_list = []
        for sig, (_, metasig) in zip(sigs, iter_meta):
            sig_sample_rate = metasig["sample_rate"]
            new_sig = resample_poly(sig, up=self.sample_rate, down=sig_sample_rate)
            new_sigs_list.append(new_sig)

        metainfo["original_sample_rate"] = metainfo["sample_rate"]
        metainfo["sample_rate"] = self.sample_rate

        # Redefine
        data["metainfo"] = metainfo
        data["signal"] = np.array(new_sigs_list)

        return data


class Resize2D(Transform):
    """Resize transform for image-like data

    Args:
        width: Output width
        height: Output height
        interpolation: interpolation method
    """

    def __init__(
        self,
        width: int,
        height: int,
        interpolation: Literal["nearest", "linear", "cubic", "area", "lanczos"],
    ):
        super().__init__()

        interpolations: Dict[str, int] = {
            "nearest": cv.INTER_NEAREST,
            "linear": cv.INTER_LINEAR,
            "cubic": cv.INTER_CUBIC,
            "area": cv.INTER_AREA,
            "lanczos": cv.INTER_LANCZOS4,
        }

        self.width = width
        self.height = height
        self.interpolation = interpolations[interpolation]

    def transform(self, data):
        dsize = (self.width, self.height)

        data = data.copy()
        metainfo = data["metainfo"].copy(deep=True)
        signals = data["signal"]

        ret = []
        for (_, entry), img in zip(metainfo.iterrows(), signals):
            resized_image = cv.resize(img, dsize, self.interpolation)
            ret.append(resized_image)

        data["signal"] = np.array(ret)
        return data
