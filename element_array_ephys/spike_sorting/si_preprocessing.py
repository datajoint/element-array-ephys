import spikeinterface as si
from spikeinterface import preprocessing


def CatGT(recording):
    recording = si.preprocessing.phase_shift(recording)
    recording = si.preprocessing.common_reference(
        recording, operator="median", reference="global"
    )
    return recording


def IBLdestriping(recording):
    # From International Brain Laboratory. “Spike sorting pipeline for the International Brain Laboratory”. 4 May 2022. 9 Jun 2022.
    recording = si.preprocessing.highpass_filter(recording, freq_min=400.0)
    bad_channel_ids, channel_labels = si.preprocessing.detect_bad_channels(recording)
    # For IBL destriping interpolate bad channels
    recording = si.preprocessing.interpolate_bad_channels(bad_channel_ids)
    recording = si.preprocessing.phase_shift(recording)
    # For IBL destriping use highpass_spatial_filter used instead of common reference
    recording = si.preprocessing.highpass_spatial_filter(
        recording, operator="median", reference="global"
    )
    return recording


def IBLdestriping_modified(recording):
    # From SpikeInterface Implementation (https://spikeinterface.readthedocs.io/en/latest/how_to/analyse_neuropixels.html)
    recording = si.preprocessing.highpass_filter(recording, freq_min=400.0)
    bad_channel_ids, channel_labels = si.preprocessing.detect_bad_channels(recording)
    # For IBL destriping interpolate bad channels
    recording = recording.remove_channels(bad_channel_ids)
    recording = si.preprocessing.phase_shift(recording)
    recording = si.preprocessing.common_reference(
        recording, operator="median", reference="global"
    )
    return recording
