from abc import abstractmethod
from typing import Dict, List, Literal

import cv2 as cv
import numpy as np
import pandas as pd
import essentia.standard
from scipy import interpolate
from sklearn import preprocessing
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
from scipy.signal import spectrogram, resample_poly
from sklearn.base import BaseEstimator, TransformerMixin
import copy

from vibdata.deep.signal.core import SignalSample

class Filter(BaseEstimator, TransformerMixin):
    @abstractmethod
    def filter(self, data):
        pass

    def fit(self, *args, **kwargs):
        return self

    def __call__(self, data):
        return self.filter(data)

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


class FilterByValue(Filter):
    def __init__(self, on_field, values) -> None:
        super().__init__()
        self.on_field = on_field
        if isinstance(values, str):
            self.values = [values]
        elif hasattr(values, "__iter__") or hasattr(values, "__getitem__"):
            self.values = values
        else:
            self.values = [values]

    def filter(self, data):
        D = data["metainfo"][self.on_field]
        D = D.isin(self.values)
        if D.any():
            return data
        return None


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

class SequentialFilter(Filter):
    def __init__(self, filters: List[Filter]):
        super().__init__()
        self.filters = filters

    def filter(self, data):
        for f in self.filters:
            if f is None:
                continue
            if hasattr(f, "filter"):
                data = f.filter(data)
            else:
                data = f(data)
            if data is None:
                return None
        return data

    def append(self, other: Filter) -> None:
        self.filters.append(other)

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


class FeatureExtractor(Transform):
    def __init__(self, features: list[Transform] | list[str]) -> None:
        super().__init__()
        if all(isinstance(t, Transform) for t in features):
            self.features = features
        elif all(isinstance(t, str) for t in features):
            gdict = globals()
            self.features = [gdict[t]() for t in features]
        else:
            raise ValueError("features must be a list of transforms or list of valid strings")

    def transform(self, data):
        new_data = data.copy()
        new_signals = np.empty((len(data["metainfo"]), len(self.features)))
        for i, feature in enumerate(self.features):
            feat_data = np.array(
                [
                    feature.transform({"metainfo": metainfo, "signal": signal})
                    for (_, metainfo), signal in zip(data["metainfo"].iterrows(), data["signal"])
                ]
            )
            new_signals[:, i] = feat_data

        new_data["signal"] = new_signals
        return new_data

class Aggregator(Transform):
    """
    Aggregate features from multiple transform pipelines into a single 1D feature vector per sample.

    Each transform in `pipelines` must be a Transform (e.g., Sequential) that outputs a dict with:
        - "signal": np.ndarray of shape (n_samples, n_features_i)
        - "metainfo": pd.DataFrame with metadata for each sample.

    The Aggregator concatenates all resulting "signal" arrays along the feature axis (axis=1),
    assuming they have the same number of rows (samples).

    Example:
    --------
    features_funcs = [Kurtosis(), Skewness(), RootMeanSquare()]
    transforms = Aggregator(
        Sequential([
            SplitSampleRate(),
            FeatureExtractor(features=features_funcs),
        ]),
        Sequential([
            SplitSampleRate(),
            FFT(),
            FeatureExtractor(features=features_funcs),
        ]),
    )
    """

    def __init__(self, pipelines: List[Transform]):
        super().__init__()
        self.pipelines = pipelines

    def transform(self, data):
        # Apply each pipeline to a *copy* of the input data
        all_features = []
        metainfo_ref = None

        for pipeline in self.pipelines:
            out = pipeline.transform(copy.deepcopy(data))

            if not isinstance(out, dict) or "signal" not in out or "metainfo" not in out:
                raise ValueError(f"Pipeline {pipeline} must return a dict with 'signal' and 'metainfo'.")

            signal = np.asarray(out["signal"])

            # Initialize metainfo reference
            if metainfo_ref is None:
                metainfo_ref = out["metainfo"].reset_index(drop=True)
                n_samples = len(signal)
            else:
                # Ensure same number of samples
                if len(signal) != n_samples:
                    raise ValueError(
                        f"All pipelines must produce the same number of samples. "
                        f"Got {len(signal)} and expected {n_samples}."
                    )

            all_features.append(signal)

        # Concatenate along feature axis
        aggregated_features = np.concatenate(all_features, axis=1)

        return {"signal": aggregated_features, "metainfo": metainfo_ref}



class Kurtosis(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        return kurtosis(signal)


class RootMeanSquare(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        return np.sqrt(sum(np.square(signal)) / len(signal))


class StandardDeviation(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        return np.std(signal)


class Mean(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        return np.mean(signal)


class LogAttackTime(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        sample_rate = data["metainfo"]["sample_rate"]
        signal = data["signal"]

        sample_rate_float32 = np.float32(sample_rate)
        envelope = essentia.standard.Envelope(sampleRate=sample_rate_float32, applyRectification=False)
        signal_envelope = envelope(signal.astype("float32"))

        logAttackTime = essentia.standard.LogAttackTime(sampleRate=sample_rate_float32)
        return logAttackTime(signal_envelope)[0]


class TemporalDecrease(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        sample_rate = data["metainfo"]["sample_rate"]
        signal = data["signal"]

        decrease = essentia.standard.Decrease(range=((len(signal.astype("float32")) - 1) / np.float32(sample_rate)))
        return decrease(signal.astype("float32"))


class TemporalCentroid(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        sample_rate = data["metainfo"]["sample_rate"]
        signal = data["signal"]

        sample_rate_float32 = np.float32(sample_rate)
        envelope = essentia.standard.Envelope(sampleRate=sample_rate_float32, applyRectification=False)
        signal_envelope = envelope(signal.astype("float32"))

        centroid = essentia.standard.Centroid(range=((len(signal.astype("float32")) - 1) / sample_rate_float32))
        return centroid(signal_envelope)


class EffectiveDuration(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        sample_rate = data["metainfo"]["sample_rate"]
        signal = data["signal"]

        sample_rate_float32 = np.float32(sample_rate)
        envelope = essentia.standard.Envelope(sampleRate=sample_rate_float32, applyRectification=False)
        signal_envelope = envelope(signal.astype("float32"))

        effective = essentia.standard.EffectiveDuration(sampleRate=sample_rate_float32)
        return effective(signal_envelope)


class ZeroCrossingRate(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]

        zeroCrossingRate = essentia.standard.ZeroCrossingRate()
        return zeroCrossingRate(signal.astype("float32"))

class PeakValue(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        return np.max(np.abs(data["signal"]))
    
class CrestFactor(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        rms = np.sqrt(sum(np.square(signal)) / len(signal))
        if rms == 0:
            return 0
        return np.max(np.abs(signal)) / rms
    
class Skewness(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        return skew(signal)

class ClearanceFactor(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        peak_value = np.max(np.abs(signal))
        base = sum(np.sqrt(np.abs(signal))) / len(signal)
        return peak_value / (base**2)
    
class ImpulseFactor(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        peak_value = np.max(np.abs(signal))
        mean_value = np.mean(np.abs(signal))
        if mean_value == 0:
            return 0
        return peak_value / mean_value

class ShapeFactor(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        rms = np.sqrt(sum(np.square(signal)) / len(signal))
        mean_value = np.mean(np.abs(signal))
        if mean_value == 0:
            return 0
        return rms / mean_value

class UpperBoundValueHistogram(Transform):
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    def transform(self, data):
        signal = data["signal"]
        max = np.max(signal)
        min = np.min(signal)
        return max + ( (0.5 * (max - min) ) / (len(signal) - 1) )
    
class LowerBoundValueHistogram(Transform):
    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    def transform(self, data):
        signal = data["signal"]
        max = np.max(signal)
        min = np.min(signal)
        return min + ( (0.5 * (max - min) ) / (len(signal) - 1) )

class Variance(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        return np.var(signal)
    
class PeakToPeak(Transform):
    def __init__(self):
        super().__init__()

    def transform(self, data):
        signal = data["signal"]
        return np.ptp(signal)  # Peak to peak value