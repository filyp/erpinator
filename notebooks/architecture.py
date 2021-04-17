import pywt
import scipy
import scipy.stats
import numpy as np
import pandas as pd
import cesium.featurize

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy.signal import butter, lfilter

from utils import cwt

timepoints_count = 181
signal_frequency = 256


def std_signal(t, m, e):
    return np.std(m)


def abs_diffs_signal(t, m, e):
    return np.sum(np.abs(np.diff(m)))


def mean_energy_signal(t, m, e):
    return np.mean(m ** 2)


def skew_signal(t, m, e):
    return scipy.stats.skew(m)


def mean_signal(t, m, e):
    return np.mean(m)


def peak_ind(t, m, e):
    return np.argmax(np.abs(m))


def peak_value(t, m, e):
    ind = np.argmax(np.abs(m))
    return m[ind]


shape_features = {
    "std": std_signal,
    "abs_diffs": abs_diffs_signal,
}

peak_features = {
    "peak_ind": peak_ind,
    "peak_value": peak_value,
}


class LowpassFilter(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        fs = signal_frequency
        cutoff = 45  # Hz
        B, A = butter(
            6, cutoff / (fs / 2), btype="low", analog=False
        )  # 6th order Butterworth low-pass

        filtered_epochs_per_channel = []
        for channel in X:
            filtered_epochs = np.array(
                [lfilter(B, A, epoch, axis=0) for epoch in channel]
            )
            filtered_epochs_per_channel.append(filtered_epochs)
        filtered_epochs_per_channel = np.array(filtered_epochs_per_channel)
        return filtered_epochs_per_channel


class ChannelExtraction(TransformerMixin, BaseEstimator):
    def __init__(self, channel_list):
        super().__init__()
        self.channel_list = channel_list

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        epochs_per_channels = np.transpose(X, (1, 0, 2))
        epochs_per_selected_channels = []

        for channel in self.channel_list:
            this_data = epochs_per_channels[channel]
            epochs_per_selected_channels.append(this_data)

        epochs_per_selected_channels = np.array(epochs_per_selected_channels)
        selected_channels_per_epoch = np.transpose(
            epochs_per_selected_channels, (1, 0, 2)
        )
        #         print(f"EXTRACTION {selected_channels_per_epoch.shape}")
        return selected_channels_per_epoch


# swap channels and epochs axes: from epoch_channel_timepoints to channel_epoch_timepoints and vice versa
class ChannelDataSwap(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data_channel_swaped = np.transpose(X, (1, 0, 2))
        return data_channel_swaped


class BinTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, step):
        super().__init__()
        self.step = step

    def bin_epoch(self, epoch):
        new_channels = []
        for channel in epoch:
            bins_channel = []
            index = 0
            while index + self.step < len(channel):
                this_bin = np.mean(channel[index : index + self.step])
                bins_channel.append(this_bin)
                index += self.step
            new_channels.append(bins_channel)
        return new_channels

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        binned_data = np.array([self.bin_epoch(epoch) for epoch in X])
        return binned_data


class IcaPreprocessing(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        timepoints_per_channel = np.concatenate(X, axis=1)
        return timepoints_per_channel.T


class IcaPostprocessing(TransformerMixin, BaseEstimator):
    def __init__(self, timepoints_count):
        super().__init__()
        self.timepoints_count = timepoints_count

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ica_transposed = X.T
        ica_n_components = X.shape[1]

        epochs_count = int(X_ica_transposed.shape[1] / timepoints_count)
        data_per_channel = X_ica_transposed.reshape(
            ica_n_components, epochs_count, timepoints_count
        )
        return data_per_channel


class Cwt(TransformerMixin, BaseEstimator):
    def __init__(self, mwt="morl", cwt_density=2, cwt_octaves=6):
        # for octaves=6, the highest frequency is 45.25 Hz
        super().__init__()
        self.mwt = mwt
        self.cwt_density = cwt_density
        self.cwt_octaves = cwt_octaves

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cwt_per_channel = []
        for data in X:
            data_cwt = np.array(
                [
                    cwt(epoch, self.mwt, self.cwt_density, self.cwt_octaves)
                    for epoch in data
                ]
            )
            cwt_per_channel.append(data_cwt)
        cwt_per_channel = np.array(cwt_per_channel)
        return cwt_per_channel


# I commented out the implementation below because I split it and I'm using IcaPostprocessing+Cwt

# class Cwt(TransformerMixin, BaseEstimator):
#     def __init__(self, timepoints_count, mwt="morl", cwt_density=2, cwt_octaves=7):
#         # for octaves=6, the highest frequency is 45.25 Hz
#         super().__init__()
#         self.timepoints_count = timepoints_count
#         self.mwt = mwt
#         self.cwt_density = cwt_density
#         self.cwt_octaves = cwt_octaves

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X_ica_transposed = X.T
#         ica_n_components = X.shape[1]

#         epochs_count = int(X_ica_transposed.shape[1] / self.timepoints_count)
#         data_per_channel = X_ica_transposed.reshape(
#             ica_n_components, epochs_count, self.timepoints_count
#         )

#         cwt_per_channel = []
#         for data in data_per_channel:
#             data_cwt = np.array(
#                 [
#                     cwt(epoch, self.mwt, self.cwt_density, self.cwt_octaves)
#                     for epoch in data
#                 ]
#             )
#             cwt_per_channel.append(data_cwt)
#         cwt_per_channel = np.array(cwt_per_channel)
#         return cwt_per_channel


class CwtFeatureVectorizer(TransformerMixin, BaseEstimator):
    def __init__(self, feature_dict):
        super().__init__()
        self.feature_dict = feature_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectorized_data = []
        for data_cwt in X:
            # cesium functions
            feature_set_cwt = cesium.featurize.featurize_time_series(
                times=None,
                values=data_cwt,
                errors=None,
                features_to_use=list(self.feature_dict.keys()),
                custom_functions=self.feature_dict,
            )
            features_per_epoch = feature_set_cwt.to_numpy()
            vectorized_data.append(features_per_epoch)
        vectorized_data = np.array(vectorized_data)
        return vectorized_data


class CwtPeakFinder(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        all_features = []
        for epochs_from_one_channel in X:
            channel_features = []
            for epoch in epochs_from_one_channel:
                epoch_features = []
                epoch_features += np.unravel_index(epoch.argmax(), epoch.shape)
                epoch_features += [epoch.max()]
                epoch_features += np.unravel_index(epoch.argmin(), epoch.shape)
                epoch_features += [epoch.min()]
                channel_features.append(epoch_features)
            all_features.append(channel_features)
        all_features = np.array(all_features)

        # transform it from shape ICA_COMP x EPOCH x PEAKINFO
        #                      to EPOCH x ICA_COMP x PEAKINFO
        all_features = all_features.transpose((1, 0, 2))

        # flatten into a shape EPOCH x (ICA_COMP*PEAKINFO)
        all_features = all_features.reshape(all_features.shape[0], -1)
        return all_features


# reshape data from (channels x epoch x features) to (epochs x channles x features)
# and then flatten it to (epoch x channels*features)
class PostprocessingTransformer(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectorized_data = np.stack(X, axis=1)
        epochs_per_channel_feature = vectorized_data.reshape(
            vectorized_data.shape[0], -1
        )
        return epochs_per_channel_feature


class PCAForEachChannel(TransformerMixin, BaseEstimator):
    def __init__(self, n_components=3, random_state=0):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        # X has a shape CHANNELS x EPOCHS x FREQ x TIMEPOINTS
        # or CHANNELS x EPOCHS x ...
        self.PCAs = []
        for channel in X:
            flattened = channel.reshape(channel.shape[0], -1)
            # flattened has a shape EPOCHS x (FREQ*TIMEPOINTS)
            pca = PCA(n_components=self.n_components, random_state=self.random_state)
            pca.fit(flattened)
            self.PCAs.append(pca)
        return self

    def transform(self, X, copy=True):
        features = []
        for channel, pca in zip(X, self.PCAs):
            flattened = channel.reshape(channel.shape[0], -1)
            # flattened has a shape EPOCHS x (FREQ*TIMEPOINTS)
            ch_transformed = pca.transform(flattened)
            features.append(ch_transformed)

        features = np.array(features)
        # transform it from shape ICA_COMP x EPOCH x WAVELET_COMP
        #                      to EPOCH x ICA_COMP x WAVELET_COMP
        features = features.transpose((1, 0, 2))

        # flatten features into a shape EPOCH x (ICA_COMP*WAVELET_COMP)
        features = features.reshape(features.shape[0], -1)
        return features


# fmt: off
# comments are mean AUROCs for personal error correct classification
# (for the first 5 participants) with SVR replaced with LDA

# 0.941    with cwt__mwt='mexh'
steps_parallel_pca = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("cwt", Cwt()),
    ("pca", PCAForEachChannel(random_state=0)),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

steps_simple = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("postprocessing", PostprocessingTransformer()),
    ("pca", PCA(random_state=0)),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# 0.695
steps_cwt_featurize = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("cwt", Cwt()),
    ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
    ("postprocessing", PostprocessingTransformer()),
    ("pca", PCA(random_state=0)),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# 0.745
steps_cwt_featurize_parallel_pca = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("cwt", Cwt(mwt="morl")),
    ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
    ("pca", PCAForEachChannel(random_state=0)),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# steps_2streams_peakfinding is better
# steps_2streams = [
#     ("pre_spatial_filter", IcaPreprocessing()),
#     ("spatial_filter", FastICA(random_state=0)),
#     ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
#     ("featurize", FeatureUnion([
#         ('shape', Pipeline([
#             ("cwt", Cwt(mwt="morl")),
#             ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
#             ("pca", PCAForEachChannel(n_components=3, random_state=0)),
#         ])),
#         ('peaks', Pipeline([
#             ("cwt", Cwt(mwt="mexh")),
#             ("cwt_feature", CwtFeatureVectorizer(feature_dict=peak_features)),
#             ("pca", PCAForEachChannel(n_components=3, random_state=0)),
#         ])),
#     ])),
#     ("scaler", StandardScaler()),
#     ("svr", SVR()),
# ]

# 0.869
steps_2streams_peakfinding = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("featurize", FeatureUnion([
        ('shape', Pipeline([
            ("cwt", Cwt(mwt="morl")),
            ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh")),
            ("peak_finder", CwtPeakFinder()),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# 0.853
steps_3streams = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("featurize", FeatureUnion([
        ('shape', Pipeline([
            ("cwt", Cwt(mwt="morl")),
            ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh")),
            ("peak_finder", CwtPeakFinder()),
        ])),
        ('power', Pipeline([
            ("cwt", Cwt(mwt="cmor0.5-1")),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# 0.854
steps_peaks_and_power = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("featurize", FeatureUnion([
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh")),
            ("peak_finder", CwtPeakFinder()),
        ])),
        ('power', Pipeline([
            ("cwt", Cwt(mwt="cmor0.5-1")),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# 0.867
steps_peaks = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("featurize", FeatureUnion([
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh")),
            ("peak_finder", CwtPeakFinder()),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# 0.777
steps_power = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("featurize", FeatureUnion([
        ('power', Pipeline([
            ("cwt", Cwt(mwt="cmor0.5-1")),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

steps_peaks_and_power_and_shape = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("featurize", FeatureUnion([
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh")),
            ("peak_finder", CwtPeakFinder()),
        ])),
        ('power', Pipeline([
            ("cwt", Cwt(mwt="cmor0.5-1")),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
        ('shape', Pipeline([
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# 0.919
steps_peaks_and_shape = [
    ("pre_spatial_filter", IcaPreprocessing()),
    ("spatial_filter", FastICA(random_state=0)),
    ("post_spatial_filter", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("featurize", FeatureUnion([
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh")),
            ("peak_finder", CwtPeakFinder())
        ])),
        ('shape', Pipeline([
            ("cwt", Cwt(mwt="mexh")),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]
# fmt: on

# ######
# the best pipelines so far and their scores for all participant are
# steps_parallel_pca                 0.930 ± 0.004
# steps_peaks_and_shape              0.911 ± 0.005
# steps_peaks_and_power_and_shape    0.911 ± 0.005

# classifier scores with best predictor (steps_parallel_pca)
# for all participants - personal
# LDA         0.930
# SVR C=0.1   0.922
# kNR n=11    0.887


# one model for all
# LDA         0.91
# SVR C=0.1   0.87
# kNR n=11    0.86

channels_order_list = [
    "Fp1",
    "AF7",
    "AF3",
    "F1",
    "F3",
    "F5",
    "F7",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "C1",
    "C3",
    "C5",
    "T7",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "P1",
    "P3",
    "P5",
    "P7",
    "P9",
    "PO7",
    "PO3",
    "O1",
    "Iz",
    "Oz",
    "POz",
    "Pz",
    "CPz",
    "Fpz",
    "Fp2",
    "AF8",
    "AF4",
    "AFz",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT8",
    "FC6",
    "FC4",
    "FC2",
    "FCz",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO8",
    "PO4",
    "O2",
]
channels_dict = dict(zip(channels_order_list, np.arange(1, 64, 1)))
red_box = [
    "F1",
    "Fz",
    "F2",
    "FC1",
    "FCz",
    "FC2",
    "C1",
    "Cz",
    "C2",
    "CP1",
    "CPz",
    "CP2",
    "P1",
    "Pz",
    "P2",
]
significant_channels = [channels_dict[channel] for channel in red_box]

# fmt: off
ica_bins_steps = [
    ("channels_filtering", ChannelExtraction(significant_channels)),
    ("ica_preprocessing", IcaPreprocessing()),
    # ("ica", FastICA(random_state=random_state)),
    ("spatial_filter", PCA(random_state=0)),
    ("ica_postprocessing", IcaPostprocessing(timepoints_count=timepoints_count)),
    ("lowpass_filter", LowpassFilter()),
    ("channel_data_swap", ChannelDataSwap()),
    ("binning", BinTransformer(step=12)),
    ("data_channel_swap", ChannelDataSwap()),
    ("postprocessing", PostprocessingTransformer()),
    ("scaler", StandardScaler()),
#     ("feature_selection", PCAForEachChannel(random_state=0)),
    ("feature_selection", PCA(random_state=0)),
    ("en", ElasticNet(random_state=0)),
]
# fmt: on
