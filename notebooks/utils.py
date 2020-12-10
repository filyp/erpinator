import mne
import numpy as np
import pywt
import glob
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

# Start and end of the segments
tmin, tmax = -0.1, 0.6

base_layout = dict(
    template="plotly_dark",
    xaxis_showgrid=False,
    yaxis_showgrid=False,
    margin=dict(l=20, r=20, t=20, b=20),
    dragmode="select",
    showlegend=False,
    width=900,
    height=560,
)


def get_wavelet(latency, frequency, times):
    signal_frequency = 1 / (times[1] - times[0])
    mother = pywt.ContinuousWavelet("mexh")
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


def load_epochs_from_file(file):
    """Load epochs from a header file.

    Args:
        file: path to a header file (.vhdr)

    Returns:
        mne Epochs

    """
    # Import the BrainVision data into an MNE Raw object
    raw = mne.io.read_raw_brainvision(file)

    # Read in the event information as MNE annotations
    annot_file = file[:-4] + "vmrk"
    #     annot = mne.read_annotations("../data/GNG_AA0303--Seg Response 5.vmrk")
    annot = mne.read_annotations(annot_file)

    # Add the annotations to our raw object so we can use them with the data
    raw.set_annotations(annot)

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

    # Read epochs
    epochs = mne.Epochs(
        raw=raw,
        events=merged_events,
        event_id=merged_event_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
    )

    return epochs


def load_all_epochs(test_participants=False, test_epochs=False):
    """Loads epochs for all participants.

    On default, loads a train set: chooses only 80% of participants
    and for each of them chooses 80% of epochs.
    It will choose them deterministically.

    Participants with less than 10 epochs per condition are rejected.

    If test_participants is set to True, it will load remaining 20% of participants.
    If test_epochs is set to True, it will load remaining 20% of epochs.
    Test epochs are chronologically after train epochs,
    because it reflects real usage (first callibration and then classification).

    Returns a 5D structure:
    PARTICIPANTS x [ERROR, CORRECT] x EPOCH X CHANNEL x TIMEPOINT
    and the last 3 dimensions are a numpy array.

    """
    header_files = glob.glob("../data/responses/*.vhdr")
    header_files = sorted(header_files)
    h_train, h_test = train_test_split(header_files, test_size=0.2, random_state=0)
    if test_participants:
        header_files = h_test
    else:
        header_files = h_train

    all_epochs = []
    for file in header_files:
        epochs = load_epochs_from_file(file)
        error = epochs["error_response"]._data
        correct = epochs["correct_response"]._data
        if len(error) < 10 or len(correct) < 10:
            # not enough data for this participant
            continue
        # shuffling is disabled to make sure test epochs are after train epochs
        err_train, err_test = train_test_split(error, test_size=0.2, shuffle=False)
        cor_train, cor_test = train_test_split(correct, test_size=0.2, shuffle=False)
        if test_epochs:
            all_epochs.append((err_test, cor_test))
        else:
            all_epochs.append((err_train, cor_train))

    return all_epochs


# def dist(f1, f2):
#     assert f1.shape == f2.shape
#     diff = f1 - f2
#     return np.dot(diff, diff) ** (1 / 2) * 1e6
#     # return np.dot(diff, diff) * 1e12


# def mask(array, window):
#     mapping = interp1d([tmin, tmax], [0, len(array)])
#     min_index, max_index = mapping(window)
#     return np.array(
#         [el if min_index < i < max_index else 0 for i, el in enumerate(array)]
#     )


# def band_pass(array, freq_range, sampling_freq):
#     sos = signal.butter(6, freq_range, "bandpass", fs=sampling_freq, output="sos")
#     return signal.sosfiltfilt(sos, array)


# def extract_erp(epoch, selected_chs, band_pass_range, sampling_freq, window):
#     filtered = epoch[selected_chs].mean(axis=0)
#     filtered = band_pass(filtered, band_pass_range, sampling_freq)
#     filtered = mask(filtered, window)
#     return filtered


# def load_gonogo_responses():
#     # Import the BrainVision data into an MNE Raw object
#     raw = mne.io.read_raw_brainvision("../data/GNG_AA0303--Seg Response 5.vhdr")

#     # Read in the event information as MNE annotations
#     annot = mne.read_annotations("../data/GNG_AA0303--Seg Response 5.vmrk")

#     # Add the annotations to our raw object so we can use them with the data
#     raw.set_annotations(annot)

#     # Map with response markers only
#     event_dict = {
#         "Stimulus/RE*ex*1_n*1_c_1*R*FB": 10004,
#         "Stimulus/RE*ex*1_n*1_c_1*R*FG": 10005,
#         "Stimulus/RE*ex*1_n*1_c_2*R": 10006,
#         "Stimulus/RE*ex*1_n*2_c_1*R": 10007,
#         "Stimulus/RE*ex*2_n*1_c_1*R": 10008,
#         "Stimulus/RE*ex*2_n*2_c_1*R*FB": 10009,
#         "Stimulus/RE*ex*2_n*2_c_1*R*FG": 10010,
#         "Stimulus/RE*ex*2_n*2_c_2*R": 10011,
#     }

#     # Map for merged correct/error response markers
#     merged_event_dict = {"correct_response": 0, "error_response": 1}

#     # Reconstruct the original events from Raw object
#     events, event_ids = mne.events_from_annotations(raw, event_id=event_dict)

#     # Merge correct/error response events
#     merged_events = mne.merge_events(
#         events,
#         [10004, 10005, 10009, 10010],
#         merged_event_dict["correct_response"],
#         replace_events=True,
#     )
#     merged_events = mne.merge_events(
#         merged_events,
#         [10006, 10007, 10008, 10011],
#         merged_event_dict["error_response"],
#         replace_events=True,
#     )

#     # Read epochs
#     epochs = mne.Epochs(
#         raw=raw,
#         events=merged_events,
#         event_id=merged_event_dict,
#         tmin=tmin,
#         tmax=tmax,
#         baseline=None,
#         preload=True,
#     )

#     return epochs
