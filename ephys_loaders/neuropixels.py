from datetime import datetime
import numpy as np
import pathlib
from .utils import handle_string


class Neuropixels:

    def __init__(self, root_dir):
        '''
        create neuropixels reader from 'root name' - e.g. the recording:

            /data/rec_1/npx_g0_t0.imec.ap.meta
            /data/rec_1/npx_g0_t0.imec.ap.bin
            /data/rec_1/npx_g0_t0.imec.lf.meta
            /data/rec_1/npx_g0_t0.imec.lf.bin

        would have a 'root name' of:

            /data/rec_1/npx_g0_t0.imec

        only a single recording is read/loaded via the root
        name & associated meta - no interpretation of g0_t0.imec, etc is
        performed at this layer.
        '''

        self.root_dir = root_dir

        meta_filepath = next(pathlib.Path(root_dir).glob('*.ap.meta'))
        self.npx_meta = NeuropixelsMeta(meta_filepath)

        self.root_name = meta_filepath.name.replace('.ap.meta', '')
        self._apdata = None
        self._lfdata, self._lfmeta = None, None

    @property
    def apdata(self):
        if self._apdata is not None:
            return self._apdata
        else:
            return self._read_bin(self.root_dir / (self.root_name + '.ap.bin'))

    @property
    def lfmeta(self):
        if self._lfmeta is not None:
            return self._lfmeta
        else:
            return _read_meta(self.root_dir / (self.root_name + '.lf.meta'))

    @property
    def lfdata(self):
        if self._lfdata is not None:
            return self._lfdata
        else:
            return self._read_bin(self.root_dir / (self.root_name + '.lf.bin'))

    def _read_bin(self, fname):
        nchan = self.npx_meta.meta['nSavedChans']
        dtype = np.dtype((np.uint16, nchan))
        return np.memmap(fname, dtype, 'r')

    def extract_spike_waveforms(self, spikes, channel, n_wf=500, wf_win=(-32, 32), bit_volts=1):
        """
        :param spikes: spike times (in second) to extract waveforms
        :param channel: channel (name, not indices) to extract waveforms
        :param n_wf: number of spikes per unit to extract the waveforms
        :param wf_win: number of sample pre and post a spike
        :param bit_volts: scalar required to convert int16 values into microvolts (default of 1)
        :return: waveforms (sample x channel x spike)
        """

        data = self.apdata
        channel_idx = [np.where(self.npx_meta.recording_channels == chn)[0][0] for chn in channel]

        spikes = np.round(spikes * self.npx_meta.meta['imSampRate']).astype(int)  # convert to sample

        np.random.shuffle(spikes)
        spikes = spikes[:n_wf]
        # ignore spikes at the beginning or end of raw data
        spikes = spikes[np.logical_and(spikes > wf_win[0], spikes < data.shape[0] - wf_win[-1])]
        if len(spikes) > 0:
            # waveform at each spike: (sample x channel x spike)
            spike_wfs = np.dstack([data[int(spk + wf_win[0]):int(spk + wf_win[-1]), channel_idx] for spk in spikes])
            return spike_wfs * bit_volts
        else:  # if no spike found, return NaN of size (sample x channel x 1)
            return np.full((len(range(*wf_win)), len(channel), 1), np.nan)


class NeuropixelsMeta:

    def __init__(self, meta_filepath):
        # a good processing reference: https://github.com/jenniferColonell/Neuropixels_evaluation_tools/blob/master/SGLXMetaToCoords.m

        self.fname = meta_filepath
        self.meta = _read_meta(meta_filepath)

        # Infer npx probe model (e.g. 1.0 (3A, 3B) or 2.0)
        probe_model = self.meta.get('imDatPrb_type', 1)
        if probe_model <= 1:
            if 'typeEnabled' in self.meta:
                self.probe_model = 'neuropixels 1.0 - 3A'
            elif 'typeImEnabled' in self.meta:
                self.probe_model = 'neuropixels 1.0 - 3B'
        elif probe_model == 21:
            self.probe_model = 'neuropixels 2.0 - SS'
        elif probe_model == 24:
            self.probe_model = 'neuropixels 2.0 - MS'
        else:
            self.probe_model = str(probe_model)

        # Get recording time
        self.recording_time = datetime.strptime(self.meta.get('fileCreateTime_original', self.meta['fileCreateTime']),
                                                '%Y-%m-%dT%H:%M:%S')
        self.recording_duration = self.meta['fileTimeSecs']

        # Get probe serial number - 'imProbeSN' for 3A and 'imDatPrb_sn' for 3B
        try:
            self.probe_SN = self.meta.get('imProbeSN', self.meta.get('imDatPrb_sn'))
        except KeyError:
            raise KeyError('Probe Serial Number not found in either "imProbeSN" or "imDatPrb_sn"')

        self.chanmap = self._parse_chanmap(self.meta['~snsChanMap']) if '~snsChanMap' in self.meta else None
        self.shankmap = self._parse_shankmap(self.meta['~snsShankMap']) if '~snsShankMap' in self.meta else None
        self.imroTbl = self._parse_imrotbl(self.meta['~imroTbl']) if '~imroTbl' in self.meta else None

        self.recording_channels = [c[0] for c in self.imroTbl['data']] if self.imroTbl else None

    @staticmethod
    def _parse_chanmap(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#channel-map
        Parse channel map header structure. Converts:

            '(x,y,z)(c0,x:y)...(cI,x:y),(sy0;x:y)'

        e.g:

            '(384,384,1)(AP0;0:0)...(AP383;383:383)(SY0;768:768)'

        into dict of form:

            {'shape': [x,y,z], 'c0': [x,y], ... }
        '''

        res = {}
        for u in (i.rstrip(')').split(';') for i in raw.split('(') if i != ''):
            if (len(u)) == 1:
                res['shape'] = u[0].split(',')
            else:
                res[u[0]] = u[1].split(':')

        return res

    @staticmethod
    def _parse_shankmap(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#shank-map
        Parse shank map header structure. Converts:

            '(x,y,z)(a:b:c:d)...(a:b:c:d)'

        e.g:

            '(1,2,480)(0:0:192:1)...(0:1:191:1)'

        into dict of form:

            {'shape': [x,y,z], 'data': [[a,b,c,d],...]}

        '''
        res = {'shape': None, 'data': []}

        for u in (i.rstrip(')') for i in raw.split('(') if i != ''):
            if ',' in u:
                res['shape'] = [int(d) for d in u.split(',')]
            else:
                res['data'].append([int(d) for d in u.split(':')])

        return res

    @staticmethod
    def _parse_imrotbl(raw):
        '''
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#imro-per-channel-settings
        Parse imro tbl structure. Converts:

            '(X,Y,Z)(A B C D E)...(A B C D E)'

        e.g.:

            '(641251209,3,384)(0 1 0 500 250)...(383 0 0 500 250)'

        into dict of form:

            {'shape': (x,y,z), 'data': []}
        '''
        res = {'shape': None, 'data': []}

        for u in (i.rstrip(')') for i in raw.split('(') if i != ''):
            if ',' in u:
                res['shape'] = [int(d) for d in u.split(',')]
            else:
                res['data'].append([int(d) for d in u.split(' ')])

        return res


# ============= HELPER FUNCTIONS =============

def _read_meta(meta_filepath):
    '''
    Read metadata in 'k = v' format.

    The fields '~snsChanMap' and '~snsShankMap' are further parsed into
    'snsChanMap' and 'snsShankMap' dictionaries via calls to
    Neuropixels._parse_chanmap and Neuropixels._parse_shankmap.
    '''

    res = {}
    with open(meta_filepath) as f:
        for l in (l.rstrip() for l in f):
            if '=' in l:
                try:
                    k, v = l.split('=')
                    v = handle_string(v)
                    res[k] = v
                except ValueError:
                    pass
    return res
