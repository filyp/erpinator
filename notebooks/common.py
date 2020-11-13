import mne
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

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
    height=400,
)


# def dist(f1, f2):
#     assert f1.shape == f2.shape
#     diff = f1 - f2
#     return np.dot(diff, diff) ** (1 / 2) * 1e6
#     # return np.dot(diff, diff) * 1e12


def mask(array, window):
    mapping = interp1d([tmin, tmax], [0, len(array)])
    min_index, max_index = mapping(window)
    return np.array(
        [el if min_index < i < max_index else 0 for i, el in enumerate(array)]
    )


def band_pass(array, freq_range, sampling_freq):
    sos = signal.butter(6, freq_range, "bandpass", fs=sampling_freq, output="sos")
    return signal.sosfiltfilt(sos, array)


def extract_erp(epoch, selected_chs, band_pass_range, sampling_freq, window):
    filtered = epoch[selected_chs].mean(axis=0)
    filtered = band_pass(filtered, band_pass_range, sampling_freq)
    filtered = mask(filtered, window)
    return filtered


def load_gonogo_responses():
    # Import the BrainVision data into an MNE Raw object
    raw = mne.io.read_raw_brainvision("../data/GNG_AA0303--Seg Response 5.vhdr")

    # Read in the event information as MNE annotations
    annot = mne.read_annotations("../data/GNG_AA0303--Seg Response 5.vmrk")

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
