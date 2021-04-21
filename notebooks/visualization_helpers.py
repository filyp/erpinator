import os
import math
import pickle
import inspect
import itertools
from time import time
from copy import deepcopy

import pywt
import mne
import scipy
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import xxhash
import matplotlib
import matplotlib.cm as cm
from cachier import cachier
from plotly.subplots import make_subplots
from ipywidgets import Dropdown, FloatRangeSlider, IntSlider, FloatSlider, interact
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from ipywidgets import HBox, VBox
from ipywidgets import Dropdown, FloatRangeSlider, IntSlider, FloatSlider, interact

from utils import base_layout, blue_black_red, ERROR, CORRECT

all_channels_num = 64


def plot_ica_comp(ica_comp, channel_locations, channel_names, scale=1):
    x_loc, y_loc, z_loc = channel_locations.T

    fig = go.FigureWidget()
    fig.update_layout(**base_layout)
    fig.update_layout(width=350 * scale, height=350 * scale)

    # sort by z_loc for prettier printing
    info = list(zip(z_loc, x_loc, y_loc, channel_names, ica_comp))
    info.sort()
    _, _x_loc, _y_loc, _channel_names, _component = zip(*info)

    amp = max(np.abs(_component))

    fig.add_scatter(
        x=_x_loc,
        y=_y_loc,
        # text=_channel_names,
        text=_component,
        marker_color=_component,
        mode="markers",
        marker_size=42 * scale,
        marker_colorscale=blue_black_red,
        marker_cmax=amp,
        marker_cmin=-amp,
    )
    return fig


def plot_erps_after_spatial_filter(
    spatial_filter,
    epochs,
    plot_limit,
    erp_type,
    max_amp,
    channel_index_list,
    times,
    scale=1,
):
    if len(spatial_filter) != all_channels_num:
        # the channels are cropped
        # converse spatial_filter back to original 64 channels
        # with zeroes for cropped channels
        sparse_spatial_filt = np.zeros(all_channels_num)
        for index, coef in zip(channel_index_list, spatial_filter):
            sparse_spatial_filt[index] = coef
        spatial_filter = sparse_spatial_filt

    fig = go.FigureWidget()
    fig.update_layout(**base_layout)
    fig.update_layout(
        height=350 * scale,
        width=600 * scale,
        xaxis_range=[times[0], times[-1]],
        yaxis_range=[-max_amp, max_amp],
    )

    # individual ERPs
    grouped = epochs.groupby(["id"])
    for participant_id in epochs["id"].unique()[:plot_limit]:
        df = grouped.get_group(participant_id)

        err = np.stack(df.loc[df["marker"] == ERROR]["epoch"].values)
        cor = np.stack(df.loc[df["marker"] == CORRECT]["epoch"].values)
        all_ = np.stack(df["epoch"].values)
        cor_mean = cor.mean(axis=0)
        err_mean = err.mean(axis=0)
        all_mean = all_.mean(axis=0)
        err_erp = np.tensordot(err_mean, spatial_filter, axes=([0], [0]))
        cor_erp = np.tensordot(cor_mean, spatial_filter, axes=([0], [0]))
        all_erp = np.tensordot(all_mean, spatial_filter, axes=([0], [0]))
        dif_erp = cor_erp - err_erp

        if erp_type == "correct":
            fig.add_scatter(x=times, y=cor_erp)
        elif erp_type == "error":
            fig.add_scatter(x=times, y=err_erp)
        elif erp_type == "all":
            fig.add_scatter(x=times, y=all_erp)
        elif erp_type == "difference":
            fig.add_scatter(x=times, y=dif_erp)
        else:
            raise ValueError("bad argument for erp_type")
        fig.update_traces(line_width=1)

    # ERPs averaged across participants
    err = np.stack(epochs.loc[epochs["marker"] == ERROR]["epoch"].values)
    cor = np.stack(epochs.loc[epochs["marker"] == CORRECT]["epoch"].values)
    cor_mean = cor.mean(axis=0)
    err_mean = err.mean(axis=0)
    err_erp = np.tensordot(err_mean, spatial_filter, axes=([0], [0]))
    cor_erp = np.tensordot(cor_mean, spatial_filter, axes=([0], [0]))

    if erp_type == "correct":
        fig.add_scatter(x=times, y=cor_erp, line_width=5, line_color="green")
    elif erp_type == "error":
        fig.add_scatter(x=times, y=err_erp, line_width=5, line_color="red")

    return fig


def visualize_spatial_components(
    pipeline,
    epochs,
    channel_locations,
    channel_names,
    times,
    plot_limit=200,
    erp_type=None,
    max_amp=0.0001,
    channel_index_list=None,
    scale=1,
    flip_mask=None,
):
    """
    pipeline
        sklearn Pipeline to be visualized
    epochs
        dataframe with all the epochs
    channel_locations
        x, y, z coordinates from mne data
    channel_names
        names of the electrodes
    times
        list of all timepoint values in milliseconds
    plot_limit
        how many people to plot - too high makes the plot hard to read
        only has effect if individual==True
    erp_type
        only has effect if individual==True
        possible values:
            'correct'     plot the average of all correct epochs for each person
            'error'       plot the average of all error epochs for each person
            'all'         plot the average of all the epochs for each person
            'difference'  plot the average of correct epochs minus average of error epochs for each person
    max_amp
        maximum amplitude for component plotting
    channel_index_list
        if we use only some subset of electrodes, this is the list of their indexes
    scale
        scale of the plots
    flip_mask
        optional array of 1s and -1s
        its length must be the same as the number of spatial components
        setting -1 for a corresponding spatial component flips its sign for better readability
    """
    fitted_steps = dict(pipeline.steps)
    spatial = fitted_steps["spatial_filter"]

    plots = []
    for spatial_comp_num, spatial_comp in enumerate(spatial.components_):
        if flip_mask is not None:
            spatial_comp = spatial_comp * flip_mask[spatial_comp_num]
        plots.append(
            HBox(
                [
                    plot_ica_comp(
                        spatial_comp, channel_locations, channel_names, scale=scale
                    ),
                    plot_erps_after_spatial_filter(
                        spatial_comp,
                        epochs,
                        plot_limit=plot_limit,
                        erp_type=erp_type,
                        max_amp=max_amp,
                        channel_index_list=channel_index_list,
                        times=times,
                        scale=scale,
                    ),
                ]
            )
        )
    return VBox(plots)


def plot_pca_shape(pca_comps, mwt, clf_coefs, xs, max_amp, scale=1, heatmap=False):
    fig = go.FigureWidget()
    fig.update_layout(**base_layout)
    fig.update_layout(height=350 * scale, width=600 * scale)
    if not heatmap:
        fig.update_layout(yaxis_range=[-max_amp, max_amp])
    accs = []
    for i, comp in enumerate(pca_comps):
        if mwt is not None:
            # CWT+PCA in practice, multiplies by this shape
            # TODO test this block, if this is really the shape
            comp = comp.reshape(-1, timepoints_count)
            acc = np.zeros_like(xs)
            for amps_for_freq, freq in zip(comp, get_frequencies()):
                for amp, latency in zip(amps_for_freq, xs):
                    wv = get_wavelet(latency, freq, xs, mwt)
                    acc += wv * amp
        else:
            acc = np.copy(comp)

        # weight by the component importance from classifier
        acc *= clf_coefs[i]
        if not heatmap:
            fig.add_scatter(x=xs, y=acc)
        accs.append(acc)

    # show also the sum of all pca comps weighted by importance
    acc = np.sum(accs, axis=0)
    accs.append(acc)

    if not heatmap:
        fig.add_scatter(x=xs, y=acc, line_width=5, line_color="black")
    else:
        # reverse, so that later components are on the bottom
        accs = np.array(accs[::-1])
        fig.add_heatmap(
            x=xs, z=accs, zmin=-max_amp, zmax=max_amp, colorscale=blue_black_red
        )

    return fig


def visualize_pipeline(
    pipeline,
    channel_locations,
    channel_names,
    times,
    clf_coefs_all=None,
    max_amp=0.018,
    scale=1,
    heatmap=False,
    one_pca=False,
    flip_mask=None,
):
    """
    pipeline
        sklearn Pipeline to be visualized
    channel_locations
        x, y, z coordinates from mne data
    channel_names
        names of the electrodes
    times
        list of all timepoint values in milliseconds
    clf_coefs_ll
        optional classifier coefficients to be used to weigh component plots
        if left as None, assume that lasso classifier was used, and use its coefs
    max_amp
        maximum amplitude for component plotting
    scale
        scale of the plots
    heatmap
        whether to use a heatmap instead of overlayed plots for each component
    one_pca
        whether only one PCA if fitted instead of a separate PCA for each spatial component
    flip_mask
        optional array of 1s and -1s
        its length must be the same as the number of spatial components
        setting -1 for a corresponding spatial component flips its sign for better readability
    """
    if heatmap:
        print("the component on the bottom is the sum of all the above components")

    fitted_steps = dict(pipeline.steps)
    spatial = fitted_steps["spatial_filter"]

    if "pca" in fitted_steps:
        dims_reduction = fitted_steps["pca"]
    elif "feature_selection" in fitted_steps:
        dims_reduction = fitted_steps["feature_selection"]

    if clf_coefs_all is None:
        if "lasso" in fitted_steps:
            clf_coefs_all = fitted_steps["lasso"].coef_
        elif "en" in fitted_steps:
            clf_coefs_all = fitted_steps["en"].coef_
        elif "lda" in fitted_steps:
            clf_coefs_all = fitted_steps["lda"].coef_[0]

    if "binning" in fitted_steps:
        bin_step = fitted_steps["binning"].step
        xs = times[bin_step // 2 :: bin_step]
    else:
        xs = times

    if one_pca:
        pca_comps_separated = dims_reduction.components_.reshape(
            len(dims_reduction.components_), len(spatial.components_), -1
        )
    else:
        pcas = dims_reduction.PCAs
        clf_coefs_for_each_ica_comp = clf_coefs_all.reshape(
            len(spatial.components_), -1
        )
        # the line below was tested visually to be the wrong unflattening
        # clf_coefs_for_each_ica_comp = clf_coefs_all.reshape(-1, len(ica.components_)).T

    plots = []
    for ica_comp_num, ica_comp in enumerate(spatial.components_):
        if one_pca:
            pca_comps = pca_comps_separated[:, ica_comp_num, :]
            clf_coefs = clf_coefs_all
        else:
            pca_comps = pcas[ica_comp_num].components_
            clf_coefs = clf_coefs_for_each_ica_comp[ica_comp_num]

        if "cwt" in fitted_steps:
            mwt = fitted_steps["cwt"].mwt
        else:
            mwt = None

        if flip_mask is not None:
            ica_comp = ica_comp * flip_mask[ica_comp_num]
            pca_comps = pca_comps * flip_mask[ica_comp_num]

        plots.append(
            HBox(
                [
                    plot_ica_comp(
                        ica_comp, channel_locations, channel_names, scale=scale
                    ),
                    plot_pca_shape(
                        pca_comps,
                        mwt,
                        clf_coefs,
                        xs=xs,
                        max_amp=max_amp,
                        scale=scale,
                        heatmap=heatmap,
                    ),
                ]
            )
        )
        # display(HBox([plot_ica_comp(ica_comp), plot_pca_comps_on_cwt(pca_comps)]))
    return VBox(plots)


# def visualize_pipeline_but_focus_on_pca_comps(
#     pipeline,
#     clf_coefs_all=None,
#     max_amp=0.015,
#     scale=1,
# ):
#     """
#     Note: works only with common PCA for all channels and no CWT

#     pipeline
#         sklearn Pipeline to be visualized
#     clf_coefs_ll
#         optional classifier coefficients to be used to weigh component plots
#         if left as None, assume that lasso classifier was used, and use its coefs
#     max_amp
#         maximum amplitude for component plotting
#     scale
#         scale of the plots
#     """
#     fitted_steps = dict(pipeline.steps)
#     spatial = fitted_steps["spatial_filter"]
#     pca = fitted_steps["pca"]

#     if clf_coefs_all is None:
#         # clf_coefs_all = fitted_steps["lda"].coef_[0]
#         clf_coefs_all = fitted_steps["lasso"].coef_

#     for pca_comp, clf_coef in zip(pca.components_, clf_coefs_all):
#         pca_comp_2d = pca_comp.reshape(len(spatial.components_), -1)

#         fig = go.FigureWidget()
#         fig.update_layout(**base_layout)
#         fig.update_layout(height=350 * scale, width=600 * scale)
#         fig.add_heatmap(
#             x=times,
#             z=pca_comp_2d * clf_coef,
#             zmin=-max_amp,
#             zmax=max_amp,
#             colorscale=blue_black_red,
#         )
#         display(fig)

# # for each PCA component, plot a heatmap with its components
# # with each row corresponding to one spatial filter

# split_index = 0
# visualize_pipeline_but_focus_on_pca_comps(pipelines[split_index])


def visualize_time_features_as_heatmap(
    pipeline,
    times,
    clf_coefs_all=None,
    max_amp=0.018,
    scale=1,
    one_pca=True,
    flip_mask=None,
):
    """
    pipeline
        sklearn Pipeline to be visualized
    times
        list of all timepoint values in milliseconds
    clf_coefs_ll
        optional classifier coefficients to be used to weigh component plots
        if left as None, assume that a linear classifier was used, and use its coefs
    max_amp
        maximum amplitude for component plotting
    scale
        scale of the plots
    one_pca
        whether only one PCA if fitted instead of a separate PCA for each spatial component
    flip_mask
        optional array of 1s and -1s
        its length must be the same as the number of spatial components
        setting -1 for a corresponding spatial component flips its sign for better readability
    """

    fitted_steps = dict(pipeline.steps)
    spatial = fitted_steps["spatial_filter"]

    if "pca" in fitted_steps:
        dims_reduction = fitted_steps["pca"]
    elif "feature_selection" in fitted_steps:
        dims_reduction = fitted_steps["feature_selection"]

    if clf_coefs_all is None:
        if "lasso" in fitted_steps:
            clf_coefs_all = fitted_steps["lasso"].coef_
        elif "en" in fitted_steps:
            clf_coefs_all = fitted_steps["en"].coef_
        elif "lda" in fitted_steps:
            clf_coefs_all = fitted_steps["lda"].coef_[0]

    if "binning" in fitted_steps:
        bin_step = fitted_steps["binning"].step
        xs = times[bin_step // 2 :: bin_step]
    else:
        xs = times

    if one_pca:
        pca_comps_separated = dims_reduction.components_.reshape(
            len(dims_reduction.components_), len(spatial.components_), -1
        )
    else:
        pcas = dims_reduction.PCAs
        clf_coefs_for_each_ica_comp = clf_coefs_all.reshape(
            len(spatial.components_), -1
        )
        # the line below was tested visually to be the wrong unflattening
        # clf_coefs_for_each_ica_comp = clf_coefs_all.reshape(-1, len(ica.components_)).T

    summed_time_components = []
    for ica_comp_num, ica_comp in enumerate(spatial.components_):
        if one_pca:
            pca_comps = pca_comps_separated[:, ica_comp_num, :]
            clf_coefs = clf_coefs_all
        else:
            pca_comps = pcas[ica_comp_num].components_
            clf_coefs = clf_coefs_for_each_ica_comp[ica_comp_num]

        if "cwt" in fitted_steps:
            mwt = fitted_steps["cwt"].mwt
        else:
            mwt = None

        if flip_mask is not None:
            ica_comp = ica_comp * flip_mask[ica_comp_num]
            pca_comps = pca_comps * flip_mask[ica_comp_num]

        accs = []
        for i, comp in enumerate(pca_comps):
            if mwt is not None:
                # CWT+PCA in practice, multiplies by this shape
                # TODO test this block, if this is really the shape
                comp = comp.reshape(-1, timepoints_count)
                acc = np.zeros_like(xs)
                for amps_for_freq, freq in zip(comp, get_frequencies()):
                    for amp, latency in zip(amps_for_freq, xs):
                        wv = get_wavelet(latency, freq, xs, mwt)
                        acc += wv * amp
            else:
                acc = np.copy(comp)

            # weight by the component importance from classifier
            acc *= clf_coefs[i]
            accs.append(acc)

        # show also the sum of all pca comps weighted by importance
        acc = np.sum(accs, axis=0)
        summed_time_components.append(acc)

    fig = go.FigureWidget()
    fig.update_layout(**base_layout)
    fig.update_layout(height=350 * scale, width=600 * scale)
    # reverse components for more intuitive plotting
    summed_time_components = summed_time_components[::-1]
    fig.add_heatmap(
        x=xs,
        z=summed_time_components,
        zmin=-max_amp,
        zmax=max_amp,
        colorscale=blue_black_red,
    )
    return fig
