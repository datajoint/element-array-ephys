from os import path
from datetime import datetime
import pathlib
import pandas as pd
import numpy as np
import re
import logging
from .utils import convert_to_number

log = logging.getLogger(__name__)


class Kilosort:

    kilosort_files = [
        'params.py',
        'amplitudes.npy',
        'channel_map.npy',
        'channel_positions.npy',
        'pc_features.npy',
        'pc_feature_ind.npy',
        'similar_templates.npy',
        'spike_templates.npy',
        'spike_times.npy',
        'spike_times_sec.npy',
        'spike_times_sec_adj.npy',
        'template_features.npy',
        'template_feature_ind.npy',
        'templates.npy',
        'templates_ind.npy',
        'whitening_mat.npy',
        'whitening_mat_inv.npy',
        'spike_clusters.npy',
        'cluster_groups.csv',
        'cluster_KSLabel.tsv'
    ]

    # keys to self.files, .data are file name e.g. self.data['params'], etc.
    kilosort_keys = [path.splitext(kilosort_file)[0] for kilosort_file in kilosort_files]

    def __init__(self, kilosort_dir):
        self._kilosort_dir = pathlib.Path(kilosort_dir)
        self._files = {}
        self._data = None
        self._clusters = None

        params_filepath = kilosort_dir / 'params.py'

        if not params_filepath.exists():
            raise FileNotFoundError(f'No Kilosort output found in: {kilosort_dir}')

        self._info = {'time_created': datetime.fromtimestamp(params_filepath.stat().st_ctime),
                      'time_modified': datetime.fromtimestamp(params_filepath.stat().st_mtime)}

    @property
    def data(self):
        if self._data is None:
            self._stat()
        return self._data

    @property
    def info(self):
        return self._info

    def _stat(self):
        self._data = {}
        for kilosort_filename in Kilosort.kilosort_files:
            kilosort_filepath = self._kilosort_dir / kilosort_filename

            if not kilosort_filepath.exists():
                log.debug('skipping {} - does not exist'.format(kilosort_filepath))
                continue

            base, ext = path.splitext(kilosort_filename)
            self._files[base] = kilosort_filepath

            if kilosort_filename == 'params.py':
                log.debug('loading params.py {}'.format(kilosort_filepath))
                # params.py is a 'key = val' file
                params = {}
                for line in open(kilosort_filepath, 'r').readlines():
                    k, v = line.strip('\n').split('=')
                    params[k.strip()] = convert_to_number(v.strip())
                log.debug('params: {}'.format(params))
                self._data[base] = params

            if ext == '.npy':
                log.debug('loading npy {}'.format(kilosort_filepath))
                d = np.load(kilosort_filepath, mmap_mode='r',
                            allow_pickle=False, fix_imports=False)
                self._data[base] = (np.reshape(d, d.shape[0])
                                    if d.ndim == 2 and d.shape[1] == 1 else d)

        self._data['channel_map'] = self._data['channel_map'].flatten()

        # Read the Cluster Groups
        for cluster_pattern, cluster_col_name in zip(['cluster_groups.*', 'cluster_KSLabel.*'],
                                                     ['group', 'KSLabel']):
            try:
                cluster_file = next(self._kilosort_dir.glob(cluster_pattern))
            except StopIteration:
                pass
            else:
                cluster_file_suffix = cluster_file.suffix
                assert cluster_file_suffix in ('.csv', '.tsv', '.xlsx')
                break
        else:
            raise FileNotFoundError(
                'Neither "cluster_groups" nor "cluster_KSLabel" file found!')

        if cluster_file_suffix == '.tsv':
            df = pd.read_csv(cluster_file, sep='\t', header=0)
        elif cluster_file_suffix == '.xlsx':
            df = pd.read_excel(cluster_file, engine='openpyxl')
        else:
            df = pd.read_csv(cluster_file, delimiter='\t')

        self._data['cluster_groups'] = np.array(df[cluster_col_name].values)
        self._data['cluster_ids'] = np.array(df['cluster_id'].values)

    def get_best_channel(self, unit):
        template_idx = self.data['spike_templates'][
            np.where(self.data['spike_clusters'] == unit)[0][0]]
        channel_templates = self.data['templates'][template_idx, :, :]
        max_channel_idx = np.abs(channel_templates).max(axis=0).argmax()
        max_channel = self.data['channel_map'][max_channel_idx]

        return max_channel, max_channel_idx

    def extract_spike_depths(self):
        """ Reimplemented from https://github.com/cortex-lab/spikes/blob/master/analysis/ksDriftmap.m """
        
        if 'pc_features' in self.data:
            ycoords = self.data['channel_positions'][:, 1]
            pc_features = self.data['pc_features'][:, 0, :]  # 1st PC only
            pc_features = np.where(pc_features < 0, 0, pc_features)

            # ---- compute center of mass of these features (spike depths) ----

            # which channels for each spike?
            spk_feature_ind = self.data['pc_feature_ind'][self.data['spike_templates'], :]
            # ycoords of those channels?
            spk_feature_ycoord = ycoords[spk_feature_ind]
            # center of mass is sum(coords.*features)/sum(features)
            self._data['spike_depths'] = (np.sum(spk_feature_ycoord * pc_features**2, axis=1)
                                        / np.sum(pc_features**2, axis=1))
        else:
            self._data['spike_depths'] = None

        # ---- extract spike sites ----
        max_site_ind = np.argmax(np.abs(self.data['templates']).max(axis=1), axis=1)
        spike_site_ind = max_site_ind[self.data['spike_templates']]
        self._data['spike_sites'] = self.data['channel_map'][spike_site_ind]


def extract_clustering_info(cluster_output_dir):
    creation_time = None

    phy_curation_indicators = ['Merge clusters', 'Split cluster', 'Change metadata_group']
    # ---- Manual curation? ----
    phylog_filepath = cluster_output_dir / 'phy.log'
    if phylog_filepath.exists():
        phylog = pd.read_fwf(phylog_filepath, colspecs=[(6, 40), (41, 250)])
        phylog.columns = ['meta', 'detail']
        curation_row = [bool(re.match('|'.join(phy_curation_indicators), str(s)))
                        for s in phylog.detail]
        is_curated = bool(np.any(curation_row))
        if creation_time is None and is_curated:
            row_meta = phylog.meta[np.where(curation_row)[0].max()]
            datetime_str = re.search('\d{2}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}', row_meta)
            if datetime_str:
                creation_time = datetime.strptime(datetime_str.group(), '%Y-%m-%d %H:%M:%S')
            else:
                creation_time = datetime.fromtimestamp(phylog_filepath.stat().st_ctime)
                time_str = re.search('\d{2}:\d{2}:\d{2}', row_meta)
                if time_str:
                    creation_time = datetime.combine(
                        creation_time.date(),
                        datetime.strptime(time_str.group(), '%H:%M:%S').time())
    else:
        is_curated = False

    # ---- Quality control? ----
    metric_filepath = cluster_output_dir / 'metrics.csv'
    is_qc = metric_filepath.exists()
    if is_qc:
        if creation_time is None:
            creation_time = datetime.fromtimestamp(metric_filepath.stat().st_ctime)

    if creation_time is None:
        spiketimes_filepath = next(cluster_output_dir.glob('spike_times.npy'))
        creation_time = datetime.fromtimestamp(spiketimes_filepath.stat().st_ctime)

    return creation_time, is_curated, is_qc
