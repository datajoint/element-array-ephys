import numpy as np
import json
import pynwb

from element_array_ephys import ephys, probe


def curated_clustering_to_nwb(curated_clustering_key, nwbfile):
    curated_clustering_key = (ephys.CuratedClustering & curated_clustering_key).fetch1('KEY')

    clustering_query = (ephys.EphysRecording
                        * ephys.ClusteringTask
                        & curated_clustering_key)

    # ---- Probe Insertion Location ----
    if ephys.InsertionLocation & clustering_query:
        insert_location = {
            k: str(v) for k, v in (ephys.InsertionLocation & clustering_query).fetch1().items()
            if k not in ephys.InsertionLocation.primary_key}
        insert_location = json.dumps(insert_location)
    else:
        insert_location = 'N/A'

    # ---- Electrode Configuration ----
    electrode_config = (probe.Probe * probe.ProbeType * probe.ElectrodeConfig
                        * ephys.ProbeInsertion * ephys.EphysRecording ^ clustering_query).fetch1()
    ephys_device_name = f'{electrode_config["probe"]} ({electrode_config["probe_type"]})'
    ephys_device = nwbfile.create_device(name=ephys_device_name)
    electrode_group = nwbfile.create_electrode_group(
        name=electrode_config['electrode_config_name'],
        description=json.dumps(electrode_config, default=str),
        device=ephys_device,
        location=insert_location)

    electrode_query = probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode & electrode_config
    for additional_attribute in ['shank', 'shank_col', 'shank_row']:
        nwbfile.add_electrode_column(
            name=electrode_query.heading.attributes[additional_attribute].name,
            description=electrode_query.heading.attributes[additional_attribute].comment)

    for electrode in electrode_query.fetch(as_dict=True):
        nwbfile.add_electrode(
            id=electrode['electrode'], group=electrode_group,
            filtering='', imp=-1.,
            x=np.nan, y=np.nan, z=np.nan,
            rel_x=electrode['x_coord'], rel_y=electrode['y_coord'], rel_z=np.nan,
            shank=electrode['shank'], shank_col=electrode['shank_col'], shank_row=electrode['shank_row'],
            location=electrode_group.location)

    # ---- Units ----
    unit_query = clustering_query @ ephys.CuratedClustering.Unit

    for additional_attribute in ['cluster_quality_label', 'spike_count', 'sampling_rate',
                                 'spike_times', 'spike_sites', 'spike_depths']:
        nwbfile.add_unit_column(
            name=unit_query.heading.attributes[additional_attribute].name,
            description=unit_query.heading.attributes[additional_attribute].comment)

    for unit in unit_query.fetch(as_dict=True):
        if ephys.WaveformSet & curated_clustering_key & {'unit': unit['unit']}:
            waveform_mean = (ephys.WaveformSet.PeakWaveform
                             & curated_clustering_key
                             & {'unit': unit['unit']}).fetch1('peak_electrode_waveform')
            # compute waveform's STD across spikes from the peak electrode
            waveform_std = np.std((ephys.WaveformSet.Waveform & curated_clustering_key
                                   & {'unit': unit['unit']}
                                   & {'electrode': unit['electrode']}).fetch1('waveforms'), axis=0)
        else:
            waveform_mean = waveform_std = np.full(1, np.nan)

        nwbfile.add_unit(id=unit['unit'],
                         electrodes=np.where(nwbfile.electrodes.id.data == int(unit['electrode']))[0],
                         electrode_group=electrode_group,
                         sampling_rate=unit['sampling_rate'],
                         cluster_quality_label=unit['cluster_quality_label'],
                         spike_times=unit['spike_times'],
                         spike_sites=unit['spike_sites'],
                         spike_count=unit['spike_count'],
                         spike_depths=unit['spike_depths'],
                         waveform_mean=waveform_mean,
                         waveform_sd=waveform_std)

    # ---- Local Field Potential ----
    if ephys.LFP.Electrode & curated_clustering_key:
        ecephys_module = nwbfile.create_processing_module(name='ecephys', description='preprocessed ephys data')
        nwb_lfp = pynwb.ecephys.LFP(name=f'probe_{electrode_config["probe"]} - LFP')

        lfp_channels = (ephys.LFP.Electrode & curated_clustering_key).fetch('electrode')
        lfp_time_stamps = (ephys.LFP & curated_clustering_key).fetch1('lfp_time_stamps')

        electrode_ind, lfp_data = [], []
        for chn in np.unique(lfp_channels):
            lfp_data.append((ephys.LFP.Electrode
                             & curated_clustering_key & {'electrode': chn}).fetch1('lfp'))
            electrode_ind.append(np.where(nwbfile.electrodes.id.data == chn)[0][0])

        electrode_table_region = nwbfile.create_electrode_table_region(
            electrode_ind, 'Electrodes table region for LFP')
        lfp_data = np.vstack(lfp_data).T  # sample x channel (in uV)

        nwb_lfp.create_electrical_series(
            name='processed_electrical_series',
            data=lfp_data * 1e-6,  # convert to Volts
            electrodes=electrode_table_region,
            timestamps=lfp_time_stamps,
        )

        ecephys_module.add_data_interface(nwb_lfp)

    return nwbfile
