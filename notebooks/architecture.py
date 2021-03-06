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

from utils import cwt

timepoints_count = 181


def std_signal(t, m, e):
    return np.std(m)


def abs_diffs_signal(t, m, e):
    return np.sum(np.abs(np.diff(m)))


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


class IcaPreprocessing(TransformerMixin, BaseEstimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        timepoints_per_channel = np.concatenate(X, axis=1)
        return timepoints_per_channel.T


class Cwt(TransformerMixin, BaseEstimator):
    def __init__(self, timepoints_count, mwt="morl", cwt_density=2):
        super().__init__()
        self.timepoints_count = timepoints_count
        self.mwt = mwt
        self.cwt_density = cwt_density

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ica_transposed = X.T
        ica_n_components = X.shape[1]

        epochs_count = int(X_ica_transposed.shape[1] / self.timepoints_count)
        data_per_channel = X_ica_transposed.reshape(
            ica_n_components, epochs_count, self.timepoints_count
        )

        cwt_per_channel = []
        for data in data_per_channel:
            data_cwt = np.array(
                [cwt(epoch, self.mwt, self.cwt_density) for epoch in data]
            )
            cwt_per_channel.append(data_cwt)
        cwt_per_channel = np.array(cwt_per_channel)
        return cwt_per_channel


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
steps_parallel_pca = [
    ("ica_preprocessing", IcaPreprocessing()),
    ("ica", FastICA(random_state=0)),
    ("cwt", Cwt(timepoints_count=timepoints_count)),
    ("pca", PCAForEachChannel(random_state=0)),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

steps_cwt_featurize = [
    ("ica_preprocessing", IcaPreprocessing()),
    ("ica", FastICA(random_state=0)),
    ("cwt", Cwt(timepoints_count=timepoints_count)),
    ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
    ("postprocessing", PostprocessingTransformer()),
    ("pca", PCA(random_state=0)),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

steps_cwt_featurize_parallel_pca = [
    ("ica_preprocessing", IcaPreprocessing()),
    ("ica", FastICA(random_state=0)),
    ("cwt", Cwt(mwt="morl", timepoints_count=timepoints_count)),
    ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
    ("pca", PCAForEachChannel(random_state=0)),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

# steps_2streams_peakfinding is better
# steps_2streams = [
#     ("ica_preprocessing", IcaPreprocessing()),
#     ("ica", FastICA(random_state=0)),
#     ("featurize", FeatureUnion([
#         ('shape', Pipeline([
#             ("cwt", Cwt(mwt="morl", timepoints_count=timepoints_count)),
#             ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
#             ("pca", PCAForEachChannel(n_components=3, random_state=0)),
#         ])),
#         ('peaks', Pipeline([
#             ("cwt", Cwt(mwt="mexh", timepoints_count=timepoints_count)),
#             ("cwt_feature", CwtFeatureVectorizer(feature_dict=peak_features)),
#             ("pca", PCAForEachChannel(n_components=3, random_state=0)),
#         ])),
#     ])),
#     ("scaler", StandardScaler()),
#     ("svr", SVR()),
# ]

steps_2streams_peakfinding = [
    ("ica_preprocessing", IcaPreprocessing()),
    ("ica", FastICA(random_state=0)),
    ("featurize", FeatureUnion([
        ('shape', Pipeline([
            ("cwt", Cwt(mwt="morl", timepoints_count=timepoints_count)),
            ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh", timepoints_count=timepoints_count)),
            ("peak_finder", CwtPeakFinder()),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

steps_3streams = [
    ("ica_preprocessing", IcaPreprocessing()),
    ("ica", FastICA(random_state=0)),
    ("featurize", FeatureUnion([
        ('shape', Pipeline([
            ("cwt", Cwt(mwt="morl", timepoints_count=timepoints_count)),
            ("cwt_feature", CwtFeatureVectorizer(feature_dict=shape_features)),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh", timepoints_count=timepoints_count)),
            ("peak_finder", CwtPeakFinder()),
        ])),
        ('power', Pipeline([
            ("cwt", Cwt(mwt="cmor0.5-1", timepoints_count=timepoints_count)),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

steps_peaks_and_power = [
    ("ica_preprocessing", IcaPreprocessing()),
    ("ica", FastICA(random_state=0)),
    ("featurize", FeatureUnion([
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh", timepoints_count=timepoints_count)),
            ("peak_finder", CwtPeakFinder()),
        ])),
        ('power', Pipeline([
            ("cwt", Cwt(mwt="cmor0.5-1", timepoints_count=timepoints_count)),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]

steps_peaks_and_power_and_shape = [
    ("ica_preprocessing", IcaPreprocessing()),
    ("ica", FastICA(random_state=0)),
    ("featurize", FeatureUnion([
        ('peaks', Pipeline([
            ("cwt", Cwt(mwt="mexh", timepoints_count=timepoints_count)),
            ("peak_finder", CwtPeakFinder()),
        ])),
        ('power', Pipeline([
            ("cwt", Cwt(mwt="cmor0.5-1", timepoints_count=timepoints_count)),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
        ('shape', Pipeline([
            ("cwt", Cwt(mwt="mexh", timepoints_count=timepoints_count)),
            ("pca", PCAForEachChannel(n_components=3, random_state=0)),
        ])),
    ])),
    ("scaler", StandardScaler()),
    ("svr", SVR()),
]
# fmt: on
