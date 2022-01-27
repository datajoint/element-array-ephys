# THIS IS DEPRECATED - TO BE REMOVED

import datajoint as dj
import numpy as np
import json
import pynwb

from element_array_ephys import ephys as ephys_acute
from element_array_ephys import ephys_chronic, ephys_no_curation, probe


def curated_clusterings_to_nwb(curated_clustering_keys, nwbfile, ephys_module=None):
    """
    Generate one NWBFile object representing all ephys data
     coming from the specified "curated_clustering_keys"
    Note: the specified "curated_clustering_keys" must be all from one session

    :param curated_clustering_keys: entries of CuratedClustering table
    :param nwbfile: nwbfile object containing session meta information
    :param ephys_module: the activated ephys module calling this function
    :return: NWBFile object
    """
    if ephys_module is None:
        for ephys_module in (ephys_acute, ephys_chronic, ephys_no_curation):
            if ephys_module.schema.is_activated():
                ephys = ephys_module
                break
        else:
            raise ValueError(f'No activated ephys module found!')
    else:
        ephys = ephys_module

    # validate input
    if isinstance(curated_clustering_keys, dj.expression.QueryExpression):
        curated_clustering_keys = (ephys.CuratedClustering & curated_clustering_keys).fetch('KEY')

    assert len(ephys._linking_module.Session & curated_clustering_keys) == 1, \
        f'Multiple sessions error! The specified "curated_clustering_keys"' \
        f' must be from one session only'

    # create processing module for LFP
    ecephys_module = nwbfile.create_processing_module(name='ecephys',
                                                      description='preprocessed ephys data')

    # add additional columns to the electrodes table
    electrodes_query = probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode
    for additional_attribute in ['shank', 'shank_col', 'shank_row']:
        nwbfile.add_electrode_column(
            name=electrodes_query.heading.attributes[additional_attribute].name,
            description=electrodes_query.heading.attributes[additional_attribute].comment)

    # add additional columns to the units table
    units_query = ephys.EphysRecording * ephys.ClusteringTask @ ephys.CuratedClustering.Unit
    for additional_attribute in ['cluster_quality_label', 'spike_count', 'sampling_rate',
                                 'spike_times', 'spike_sites', 'spike_depths']:
        nwbfile.add_unit_column(
            name=units_query.heading.attributes[additional_attribute].name,
            description=units_query.heading.attributes[additional_attribute].comment)

    # iterate through curated clusterings and export units data
    for curated_clustering_key in (ephys.CuratedClustering
                                   & curated_clustering_keys).fetch('KEY'):

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
        ephys_device = (nwbfile.get_device(ephys_device_name)
                        if ephys_device_name in nwbfile.devices
                        else nwbfile.create_device(name=ephys_device_name))

        electrode_group = nwbfile.create_electrode_group(
            name=f'{electrode_config["probe"]} {electrode_config["electrode_config_name"]}',
            description=json.dumps(electrode_config, default=str),
            device=ephys_device,
            location=insert_location)

        electrode_query = (probe.ProbeType.Electrode * probe.ElectrodeConfig.Electrode
                           & electrode_config)
        for electrode in electrode_query.fetch(as_dict=True):
            nwbfile.add_electrode(
                id=electrode['electrode'], group=electrode_group,
                filtering='', imp=-1.,
                x=np.nan, y=np.nan, z=np.nan,
                rel_x=electrode['x_coord'], rel_y=electrode['y_coord'], rel_z=np.nan,
                shank=electrode['shank'], shank_col=electrode['shank_col'], shank_row=electrode['shank_row'],
                location=electrode_group.location)

        # ---- Units ----
        electrode_df = nwbfile.electrodes.to_dataframe()
        electrode_ind = electrode_df.index[electrode_df.group_name == electrode_group.name]

        unit_query = clustering_query @ ephys.CuratedClustering.Unit
        for unit in unit_query.fetch(as_dict=True):
            waveform_mean = waveform_std = np.full(1, np.nan)
            if ephys.WaveformSet & curated_clustering_key & {'unit': unit['unit']}:
                waveform_mean = (ephys.WaveformSet.PeakWaveform
                                 & curated_clustering_key
                                 & {'unit': unit['unit']}).fetch1('peak_electrode_waveform')
                # compute waveform's STD across spikes from the peak electrode
                electrode_waveforms = (ephys.WaveformSet.Waveform & curated_clustering_key
                                       & {'unit': unit['unit']}
                                       & {'electrode': unit['electrode']}).fetch1('waveforms')
                if electrode_waveforms is not None:
                    waveform_std = np.std(electrode_waveforms, axis=0)

            nwbfile.add_unit(id=unit['unit'],
                             electrodes=np.where(electrode_ind == unit['electrode'])[0],
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
            nwb_lfp = pynwb.ecephys.LFP(name=f'probe_{electrode_config["probe"]} - LFP')

            lfp_channels = (ephys.LFP.Electrode & curated_clustering_key).fetch('electrode')
            lfp_time_stamps = (ephys.LFP & curated_clustering_key).fetch1('lfp_time_stamps')

            electrode_ind, lfp_data = [], []
            for chn in set(lfp_channels):
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
