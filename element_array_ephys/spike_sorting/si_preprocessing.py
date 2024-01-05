import spikeinterface as si
from spikeinterface import preprocessing


def mimic_catGT(recording):
    recording = si.preprocessing.phase_shift(recording)
    recording = si.preprocessing.common_reference(
        recording, operator="median", reference="global"
    )
    return recording


def mimic_IBLdestriping(recording):
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


def mimic_IBLdestriping_modified(recording):
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


preprocessing_function_mapping = {
    "catGT": mimic_catGT,
    "IBLdestriping": mimic_IBLdestriping,
    "IBLdestriping_modified": mimic_IBLdestriping_modified,
}


## Example SI parameter set
"""
{'detect_threshold': 6,
 'projection_threshold': [10, 4],
 'preclust_threshold': 8,
 'car': True,
 'minFR': 0.02,
 'minfr_goodchannels': 0.1,
 'nblocks': 5,
 'sig': 20,
 'freq_min': 150,
 'sigmaMask': 30,
 'nPCs': 3,
 'ntbuff': 64,
 'nfilt_factor': 4,
 'NT': None,
 'do_correction': True,
 'wave_length': 61,
 'keep_good_only': False,
 'PreProcessing_params': {'Filter': False,
  'BandpassFilter': True,
  'HighpassFilter': False,
  'NotchFilter': False,
  'NormalizeByQuantile': False,
  'Scale': False,
  'Center': False,
  'ZScore': False,
  'Whiten': False,
  'CommonReference': False,
  'PhaseShift': False,
  'Rectify': False,
  'Clip': False,
  'BlankSaturation': False,
  'RemoveArtifacts': False,
  'RemoveBadChannels': False,
  'ZeroChannelPad': False,
  'DeepInterpolation': False,
  'Resample': False}}
"""
