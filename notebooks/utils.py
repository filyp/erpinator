import re
import glob
import os
import ast
import os.path as op
from collections import defaultdict
from copy import deepcopy

import mne
import scipy
import numpy as np
import pywt
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


# Start and end of the segments
tmin, tmax = -0.1, 0.6
signal_frequency = 256
ERROR = 0
CORRECT = 1
blue_black_red = [[0, "rgb(0,100,255)"], [0.5, "rgb(0,0,0)"], [1, "rgb(255,40,0)"]]

base_layout = dict(
    template="plotly_dark",
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    margin=dict(l=30, r=30, t=30, b=30),
    dragmode="select",
    showlegend=False,
    width=900,
    height=560,
)


def get_frequencies(density=2, octaves=7):
    # for octaves=6, the highest frequency is 45.25 Hz
    return 2 ** (np.arange(octaves, step=1 / density))


def cwt(epoch, mwt="mexh", density=2, octaves=7):
    center_wavelet_frequency = pywt.scale2frequency(mwt, [1])[0]
    const = center_wavelet_frequency * signal_frequency

    # construct scales
    scales = const / get_frequencies(density, octaves)

    # compute coeffs
    coef, freqs = pywt.cwt(
        data=epoch, scales=scales, wavelet=mwt, sampling_period=1 / signal_frequency
    )
    if "cmor" in mwt:
        # if complex Morlet, change to real
        coef = np.abs(coef)
    return coef


def get_separations(cond1, cond2):
    # compute separation across given parameters
    # TODO think if within_class equation is OK or should conditions be rescaled
    # fmt: off
    within_class_scatter = cond1.var(axis=0) * len(cond1) + \
                           cond2.var(axis=0) * len(cond2)
    # fmt: on
    joined = np.append(cond1, cond2, axis=0)
    between_class_scatter = joined.var(axis=0) * len(joined)
    return between_class_scatter / within_class_scatter


def get_wavelet(latency, frequency, times, mwt="mexh"):
    signal_frequency = 1 / (times[1] - times[0])
    mother = pywt.ContinuousWavelet(mwt)
    scale = signal_frequency / frequency
    mex, _ = mother.wavefun(length=int(scale * 4))

    center_index = int((latency - times[0]) * signal_frequency)
    left_index = center_index - len(mex) // 2
    res = np.zeros_like(times)
    if left_index < 0:
        mex = mex[-left_index:]
        start = 0
    else:
        start = left_index

    mex = mex[: len(res) - start]
    res[start : start + len(mex)] = mex
    return res


def from_np_array(array_string):
    array_string = ",".join(array_string.replace("[ ", "[").split())
    return np.array(ast.literal_eval(array_string))


def load_epochs_from_file(file, reject_bad_segments="auto", mask=None):
    """Load epochs from a header file.

    Args:
        file: path to a header file (.vhdr)
        reject_bad_segments: 'auto' | 'annot' | 'peak-to-peak'

        Whether the epochs with overlapping bad segments are rejected by default.

        'auto' means that bad segments are rejected automatically.
        'annot' rejection based on annotations and reject only channels annotated in .vmrk file as
        'bad'.
        'peak-to-peak' rejection based on peak-to-peak amplitude of channels.

        Rejected with 'annot' and 'amplitude' channels are zeroed.

    Returns:
        mne Epochs

    """
    # Import the BrainVision data into an MNE Raw object
    raw = mne.io.read_raw_brainvision(file)

    # Construct annotation filename
    annot_file = file[:-4] + "vmrk"

    # Read in the event information as MNE annotations
    annotations = mne.read_annotations(annot_file)

    # Add the annotations to our raw object so we can use them with the data
    raw.set_annotations(annotations)

    # Map with response markers only
    event_dict = {
        "Stimulus/RE*ex*1_n*1_c_1*R*FB": 10004,
        "Stimulus/RE*ex*1_n*1_c_1*R*FG": 10005,
        "Stimulus/RE*ex*1_n*1_c_2*R": 10006,
        "Stimulus/RE*ex*1_n*2_c_1*R": 10007,
        "Stimulus/RE*ex*2_n*1_c_1*R": 10008,
        "Stimulus/RE*ex*2_n*2_c_1*R*FB": 10009,
        "Stimulus/RE*ex*2_n*2_c_1*R*FG": 10010,
        "Stimulus/RE*ex*2_n*2_c_2*R": 10011,
    }

    # Map for merged correct/error response markers
    merged_event_dict = {"correct_response": 0, "error_response": 1}

    # Reconstruct the original events from Raw object
    events, event_ids = mne.events_from_annotations(raw, event_id=event_dict)

    # Merge correct/error response events
    merged_events = mne.merge_events(
        events,
        [10004, 10005, 10009, 10010],
        merged_event_dict["correct_response"],
        replace_events=True,
    )
    merged_events = mne.merge_events(
        merged_events,
        [10006, 10007, 10008, 10011],
        merged_event_dict["error_response"],
        replace_events=True,
    )

    epochs = []
    bads = []
    this_reject_by_annotation = True

    if reject_bad_segments != "auto":
        this_reject_by_annotation = False

    # Read epochs
    temp_epochs = mne.Epochs(
        raw=raw,
        events=merged_events,
        event_id=merged_event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject_by_annotation=this_reject_by_annotation,
        preload=True,
    )

    if reject_bad_segments == "annot":
        custom_annotations = get_annotations(annot_file)
        bads = get_bads_by_annotation(custom_annotations)
    elif reject_bad_segments == "peak-to-peak":
        bads = get_bads_by_peak_to_peak_amplitude(temp_epochs)
    else:
        epochs = temp_epochs
        return epochs

    if mask is None:
        epochs = clear_bads(temp_epochs, bads)
    elif len(mask) == 64:
        epochs = reject_with_mask(temp_epochs, mask, bads)
    else:
        print(
            "Given mask has wrong shape. Expected len of 64 but got {}".format(
                len(mask)
            )
        )

    return epochs


def create_df_data(
    test_participants=False,
    test_epochs=False,
    info_filename=None,
    info="all",
    personal=True,
):
    """Loads data for all participants and create DataFrame with optional additional info from given .csv file.

    On default, loads a train set: chooses only 80% of participants
    and for each of them chooses 80% of epochs.
    It will choose them deterministically.

    Participants with less than 10 epochs per condition are rejected.

    If test_participants is set to True, it will load remaining 20% of participants.
    If test_epochs is set to True, it will load remaining 20% of epochs.
    Test epochs are chronologically after train epochs,
    because it reflects real usage (first callibration and then classification).

    Parameters
    ----------
    test_participants: bool
        whether load data for training or final testing.
        If true load participants data for testing.
    test_epochs: bool
        whether load data for training or final testing.
        If true load epochs of each participants data for testing.
    info_filename: String | None
        path to .csv file with additional data.
    info: array
        listed parameters from the info file to be loaded.
        if 'all', load all parameters
    personal: bool
        whether a model will be both trained and tested on epochs from one person
        if false, person's epochs aren't split into test and train
        and people aren't rejected if they have too few epochs


    Returns
    -------
    go_nogo_data_df : pandas.DataFrame

    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    header_files_glob = os.path.join(dir_path, "../data/responses/*.vhdr")
    header_files = glob.glob(header_files_glob)

    header_files = sorted(header_files)
    go_nogo_data_df = pd.DataFrame()

    # cut 20% of data for testing
    h_train, h_test = train_test_split(header_files, test_size=0.2, random_state=0)

    if test_participants:
        header_files = h_test
    else:
        header_files = h_train

    for file in header_files:
        #  load eeg data for given participant
        participant_epochs = load_epochs_from_file(file)

        # and compute participant's id from file_name
        participant_id = re.match(r".*_(\w+).*", file).group(1)

        error = participant_epochs["error_response"]._data
        correct = participant_epochs["correct_response"]._data

        # exclude those participants who have too few samples
        if len(error) < 5 or len(correct) < 5:
            # not enough data for this participant
            continue

        if personal:
            # cut 20% of each participant's epochs for testing
            # shuffling is disabled to make sure test epochs are after train epochs
            # TODO: not sure if this step is necessary
            err_train, err_test = train_test_split(error, test_size=0.2, shuffle=False)
            cor_train, cor_test = train_test_split(
                correct, test_size=0.2, shuffle=False
            )
            if test_epochs:
                error = err_test
                correct = cor_test
            else:
                error = err_train
                correct = cor_train

        # construct dataframe for participant with: id|epoch_data|response_type|additional info...
        participant_df = create_df_from_epochs(
            participant_id, correct, error, info_filename, info
        )
        print(participant_id)
        go_nogo_data_df = go_nogo_data_df.append(participant_df, ignore_index=True)

    return go_nogo_data_df


def create_df_from_epochs(id, correct, error, info_filename, info):
    """Create df for each participant. DF structure is like: {id: String ; epoch: epoch_data ; marker: 1.0|0.0}
    1.0 means correct and 0.0 means error response.
    Default info extracted form .csv file is 'Rumination Full Scale' and participants' ids.
    With this info df structure is like:
    {id: String ; epoch: epoch_data ; marker: 1.0|0.0 ; File: id ; 'Rumination Full Scale': int}

    Parameters
    ----------
    id: String
        participant's id extracted from filename
    correct: array
        correct responses' data
    error: array
        error responses' data
    info_filename: String
        path to .csv file with additional data.
    info: array
        listed parameters from the info file to be loaded.
        if 'all', load all parameters

    Returns
    -------
    participant_df : pandas.DataFrame

    """
    participant_df = pd.DataFrame()
    info_df = pd.DataFrame()

    # get additional info from file
    if info_filename is not None:
        if info == "all":
            rumination_df = pd.read_csv(info_filename)
        else:
            rumination_df = pd.read_csv(info_filename, usecols=["File"] + info)
        info_df = (
            rumination_df.loc[rumination_df["File"] == id]
            .reset_index()
            .drop("index", axis=1)
        )

    for epoch in correct:
        epoch_df = pd.DataFrame(
            {"id": [id], "epoch": [epoch], "marker": [CORRECT]}
        ).join(info_df)
        participant_df = participant_df.append(epoch_df, ignore_index=True)

    for epoch in error:
        epoch_df = pd.DataFrame({"id": [id], "epoch": [epoch], "marker": [ERROR]}).join(
            info_df
        )
        participant_df = participant_df.append(epoch_df, ignore_index=True)

    return participant_df


def clear_bads(epochs, bads, replacement=0):
    """Clear specified channels in epochs.

    Parameters
    ----------
    epochs : mne Epochs
        epochs for clearing.
    bads: array[(epoch, channel)]
        list of tuples (epoch, channel) for clearing.
    replacement: int
        sign for replacing.

    Returns
    -------
    epochs : mne Epochs
        cleared epochs.
    """

    eeg_data = epochs.get_data()
    overlapped_epochs_set = set()

    #     print("Cleared channels: ")
    for epoch_index, channel_index in bads:
        overlapped_epochs_set.add(epoch_index)
        #         print("channel: {} , epoch_index: {}".format(channel_index, epoch_index))

        eeg_data[epoch_index][channel_index] = [replacement]

    print("Amount of overlapped epochs: {}".format(len(overlapped_epochs_set)))
    print(f"Overlapped epochs: {overlapped_epochs_set}")

    return epochs


def peak_to_peak_amplitude(signal):
    n_samples = len(signal)

    signal_fft = np.fft.fft(signal)
    amplitudes = 2 / n_samples * np.abs(signal_fft)
    peak_to_peak_amplitude = max(amplitudes) - min(amplitudes)

    #     print('peak to peak amplitude {}, max amplitude: {}'.format(peak_to_peak_amplitude, max(amplitudes)))

    return peak_to_peak_amplitude


def get_bad_epochs_channel_index(annotation, current_segment_index):
    """Gets epochs and channels indices where annotation given as parameter occures.

    Parameters
    ----------
    annotation: OrderDict
    current_segment_index: int

    Returns
    -------
    bad_epoch_channel_index: array[(epoch, channel)]

    """
    onset = annotation["onset"]  # as position in datapoint
    duration = annotation["duration"]  # as ticks
    channel_num = annotation["channel_num"]
    channel_index = channel_num - 1

    bad_epoch_channel_index = []

    bad_interval_start_index = current_segment_index
    bad_interval_end_index = get_epoch_index(onset_in_ticks=onset + duration)

    for i in range(bad_interval_start_index, bad_interval_end_index + 1):
        bad_epoch_channel_index.append((i, channel_index))

    return bad_epoch_channel_index


def get_epoch_index(onset_in_ticks):
    """
    onset_in_ticks: int
        Time elapsed since tick number 0 in ticks.
    """
    #     freq = raw.info['sfreq']
    freq = 256
    segment_duration = int((tmax - tmin) * freq)
    epoch_index = int(onset_in_ticks // segment_duration)

    return epoch_index


def get_bads_by_peak_to_peak_amplitude(epochs, amplitude=4e-5):
    """Finds bad epochs and channels, based on signal amplitude thresholds.

    Parameters
    ----------
    epochs : mne Epochs
        epochs for clearing.
    amplitude: float
        maximum acceptable peak-to-peak amplitude.

    Returns
    -------
    bads : array[(epoch, channel)]
    """
    epoch_data = epochs.get_data()
    bads = []

    for epoch_index in range(len(epoch_data)):
        epoch = epoch_data[epoch_index]
        for channel_index in range(len(epoch)):
            channel_data = epoch[channel_index]
            this_amplitude = peak_to_peak_amplitude(channel_data)

            if this_amplitude > amplitude:
                # print('Epoch: {}, Channel: {}, Amplitude: {}'.format(epoch_index, channel_index, this_amplitude))
                bads.append((epoch_index, channel_index))

    return bads


def get_bads_by_annotation(
    annotations, rejected_description="Bad Interval/Bad Amplitude"
):
    """Finds bad epochs and channels, based on annotation file.

    Parameters
    ----------
    epochs : mne Epochs
        epochs for clearing.
    annotations: array[OrderDict]
        list of all annotations from .vmrk file
    rejected_description: String

    Returns
    -------
    bads : array[(epoch, channel)]
    """

    bads = []
    current_segment_index = -1

    for annot in annotations:
        if annot["description"] == "New Segment/":
            current_segment_index += 1

        if annot["description"] == rejected_description:
            this_bad = get_bad_epochs_channel_index(annot, current_segment_index)
            bads = bads + this_bad

    return bads


def get_epoch_channels_dict(bads):
    """Create default dictionary from array of tuples

    Parameters
    ----------
    bads: array
        array consisting of tuples (epoch_index, channel_index) which are concernd as bad.
    Returns
    -------
    epoch_channel_dict: dict
    """

    epoch_channels_dict = defaultdict(list)

    for epoch_index, channel_index in bads:
        epoch_channels_dict[epoch_index].append(channel_index)

    return epoch_channels_dict


def reject_with_mask(epochs, mask, bads):
    """
    Parameters
    ----------
    epochs: mne Epochs
    mask: array
        array consisting of 0s and 1s determinig which channels are off and on.
        Length of mask must be equal to amount of eeg channels.
    bads: array
        array consisting of tuples (epoch_index, channel_index) which are concernd as bad.

    Returns
    -------
    epochs: mne Epochs
    """
    bads_dict = get_epoch_channels_dict(bads)
    epoch_drop_indices = []
    print(bads_dict)

    for epoch_index in bads_dict:
        channels = bads_dict.get(epoch_index)
        filtered_channels = list(filter(lambda item: mask[item] == 0, channels))

        if len(channels) != len(filtered_channels):
            epoch_drop_indices.append(epoch_index)

    epochs.drop(indices=epoch_drop_indices)

    return epochs


def _read_vmrk(fname):
    """Read annotations from a vmrk file.

    Parameters
    ----------
    fname : str
        vmrk file to be read.
    Returns
    -------
    onset : array, shape (n_annots,)
        The onsets in ticks.
    duration : array, shape (n_annots,)
        The duration in ticks.
    description : array, shape (n_annots,)
        The description of each annotation.
    channel_num : array, shape (n_annots,)
        The channel number.
    """
    # read vmrk file
    with open(fname, "rb") as fid:
        txt = fid.read()

    # we don't actually need to know the coding for the header line.
    # the characters in it all belong to ASCII and are thus the
    # same in Latin-1 and UTF-8
    header = txt.decode("ascii", "ignore").split("\n")[0].strip()
    #     _check_bv_version(header, 'marker')

    # although the markers themselves are guaranteed to be ASCII (they
    # consist of numbers and a few reserved words), we should still
    # decode the file properly here because other (currently unused)
    # blocks, such as that the filename are specifying are not
    # guaranteed to be ASCII.

    try:
        # if there is an explicit codepage set, use it
        # we pretend like it's ascii when searching for the codepage
        cp_setting = re.search(
            "Codepage=(.+)", txt.decode("ascii", "ignore"), re.IGNORECASE & re.MULTILINE
        )
        codepage = "utf-8"
        if cp_setting:
            codepage = cp_setting.group(1).strip()
        # BrainAmp Recorder also uses ANSI codepage
        # an ANSI codepage raises a LookupError exception
        # python recognize ANSI decoding as cp1252
        if codepage == "ANSI":
            codepage = "cp1252"
        txt = txt.decode(codepage)
    except UnicodeDecodeError:
        # if UTF-8 (new standard) or explicit codepage setting fails,
        # fallback to Latin-1, which is Windows default and implicit
        # standard in older recordings
        txt = txt.decode("latin-1")

    # extract Marker Infos block
    m = re.search(r"\[Marker Infos\]", txt, re.IGNORECASE)
    if not m:
        return np.array(list()), np.array(list()), np.array(list()), ""

    mk_txt = txt[m.end() :]
    m = re.search(r"^\[.*\]$", mk_txt)
    if m:
        mk_txt = mk_txt[: m.start()]

    # extract event information
    items = re.findall(r"^Mk\d+=(.*)", mk_txt, re.MULTILINE)
    onset, duration, description, channel_num = list(), list(), list(), list()
    date_str = ""
    for info in items:
        info_data = info.split(",")
        mtype, mdesc, this_onset, this_duration, this_channel_num = info_data[:5]
        # commas in mtype and mdesc are handled as "\1". convert back to comma
        mtype = mtype.replace(r"\1", ",")
        mdesc = mdesc.replace(r"\1", ",")
        if date_str == "" and len(info_data) == 5 and mtype == "New Segment":
            # to handle the origin of time and handle the presence of multiple
            # New Segment annotations. We only keep the first one that is
            # different from an empty string for date_str.
            date_str = info_data[-1]

        this_duration = int(this_duration) if this_duration.isdigit() else 0
        duration.append(this_duration)
        onset.append(int(this_onset) - 1)  # BV is 1-indexed, not 0-indexed
        description.append(mtype + "/" + mdesc)
        channel_num.append(int(this_channel_num))

    return (
        np.array(onset),
        np.array(duration),
        np.array(description),
        np.array(channel_num),
    )


def get_annotations(fname):
    """
    Returns
    -------
    annotations: array[OrderDict]

    """
    annotations_attributes = ["onset", "duration", "description", "channel_num"]

    onset, duration, description, channel_num = _read_vmrk("../data/" + fname)
    annotations_list = list(zip(onset, duration, description, channel_num))

    annotations = []

    for item in annotations_list:
        #         annot = CustomAnnotations(*item)
        annot = dict(zip(annotations_attributes, list(item)))
        annotations.append(annot)

    return annotations


def custom_gridsearch(X, y, steps, cv, regressor_params, memory):
    print("AUROC   corr     r2")

    # get params randomly
    all_params = list(ParameterGrid(regressor_params))
    # shuffle(all_params)

    for params in all_params:
        pipelines = []
        scores = []
        kf = KFold(n_splits=cv)
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            pipeline = Pipeline(deepcopy(steps), memory=memory)
            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)

            if type(steps[-1][1]) == LinearDiscriminantAnalysis:
                y_pred = pipeline.predict_proba(X_test)[:, 1]
            else:
                y_pred = pipeline.predict(X_test)
            corr = np.corrcoef(y_test, y_pred)[0][1]
            r2 = r2_score(y_test, y_pred)
            auroc = roc_auc_score(y_test, y_pred)  # it's different in classification!

            scores.append([auroc, corr, r2])
            print(f"{auroc:.3f}  {corr:.3f}  {r2:.3f}")

            pipelines.append(pipeline)

        # print scores
        print(f"{str(params):126}")
        means = np.mean(scores, axis=0)
        sems = scipy.stats.sem(scores, axis=0)
        for mean, sem in zip(means, sems):
            print(f"{mean:5.2f}±{sem:4.2f}", end="   ")
        print()

    # note that it returns pipelines only for last parameters in the grid
    return pipelines
