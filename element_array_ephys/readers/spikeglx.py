import logging
import pathlib
from datetime import datetime

import numpy as np

from .utils import convert_to_number

logger = logging.getLogger(__name__)

AP_GAIN = 80  # For NP 2.0 probes; APGain = 80 for all AP (LF is computed from AP)

# Imax values for different probe types - see metaguides (http://billkarsh.github.io/SpikeGLX/#metadata-guides)
IMAX = {
    "neuropixels 1.0 - 3A": 512,
    "neuropixels 1.0 - 3B": 512,
    "neuropixels 2.0 - SS": 8192,
    "neuropixels 2.0 - MS": 8192,
}


class SpikeGLX:
    def __init__(self, root_dir):
        """
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
        """
        self._apmeta, self._ap_timeseries = None, None
        self._lfmeta, self._lf_timeseries = None, None

        self.root_dir = pathlib.Path(root_dir)

        try:
            meta_filepath = next(pathlib.Path(root_dir).glob("*.ap.meta"))
        except StopIteration:
            raise FileNotFoundError(f"No SpikeGLX file (.ap.meta) found at: {root_dir}")

        self.root_name = meta_filepath.name.replace(".ap.meta", "")

    @property
    def apmeta(self):
        if self._apmeta is None:
            self._apmeta = SpikeGLXMeta(self.root_dir / (self.root_name + ".ap.meta"))
        return self._apmeta

    @property
    def ap_timeseries(self):
        """
        AP data: (sample x channel)
        Data are stored as np.memmap with dtype: int16
        - to convert to microvolts, multiply with self.get_channel_bit_volts('ap')
        """
        if self._ap_timeseries is None:
            self.validate_file("ap")
            self._ap_timeseries = self._read_bin(
                self.root_dir / (self.root_name + ".ap.bin")
            )
        return self._ap_timeseries

    @property
    def lfmeta(self):
        if self._lfmeta is None:
            self._lfmeta = SpikeGLXMeta(self.root_dir / (self.root_name + ".lf.meta"))
        return self._lfmeta

    @property
    def lf_timeseries(self):
        """
        LFP data: (sample x channel)
        Data are stored as np.memmap with dtype: int16
        - to convert to microvolts, multiply with self.get_channel_bit_volts('lf')
        """
        if self._lf_timeseries is None:
            self.validate_file("lf")
            self._lf_timeseries = self._read_bin(
                self.root_dir / (self.root_name + ".lf.bin")
            )
        return self._lf_timeseries

    def get_channel_bit_volts(self, band="ap"):
        """
        Extract the recorded AP and LF channels' int16 to microvolts - no Sync (SY) channels
        Following the steps specified in: https://billkarsh.github.io/SpikeGLX/Support/SpikeGLX_Datafile_Tools.zip
                dataVolts = dataInt * Vmax / Imax / gain
        """
        vmax = float(self.apmeta.meta["imAiRangeMax"])

        if band == "ap":
            imax = IMAX[self.apmeta.probe_model]
            imroTbl_data = self.apmeta.imroTbl["data"]
            imroTbl_idx = 3
            chn_ind = self.apmeta.get_recording_channels_indices(exclude_sync=True)

        elif band == "lf":
            imax = IMAX[self.lfmeta.probe_model]
            imroTbl_data = self.lfmeta.imroTbl["data"]
            imroTbl_idx = 4
            chn_ind = self.lfmeta.get_recording_channels_indices(exclude_sync=True)
        else:
            raise ValueError(f'Unsupported band: {band} - Must be "ap" or "lf"')

        # extract channels' gains
        if "imDatPrb_dock" in self.apmeta.meta:
            # NP 2.0; APGain = 80 for all AP (LF is computed from AP)
            chn_gains = [AP_GAIN] * len(imroTbl_data)
        else:
            # 3A, 3B1, 3B2 (NP 1.0)
            chn_gains = [c[imroTbl_idx] for c in imroTbl_data]

        chn_gains = np.array(chn_gains)[chn_ind]

        return vmax / imax / chn_gains * 1e6  # convert to uV as well

    def _read_bin(self, fname):
        nchan = self.apmeta.meta["nSavedChans"]
        dtype = np.dtype((np.int16, nchan))
        return np.memmap(fname, dtype, "r")

    def extract_spike_waveforms(self, spikes, channel_ind, n_wf=500, wf_win=(-32, 32)):
        """
        :param spikes: spike times (in second) to extract waveforms
        :param channel_ind: channel indices (of shankmap) to extract the waveforms from
        :param n_wf: number of spikes per unit to extract the waveforms
        :param wf_win: number of sample pre and post a spike
        :return: waveforms (in uV) - shape: (sample x channel x spike)
        """
        channel_bit_volts = self.get_channel_bit_volts("ap")[channel_ind]

        data = self.ap_timeseries

        spikes = np.round(spikes * self.apmeta.meta["imSampRate"]).astype(
            int
        )  # convert to sample
        # ignore spikes at the beginning or end of raw data
        spikes = spikes[
            np.logical_and(spikes > -wf_win[0], spikes < data.shape[0] - wf_win[-1])
        ]

        np.random.shuffle(spikes)
        spikes = spikes[:n_wf]
        if len(spikes) > 0:
            # waveform at each spike: (sample x channel x spike)
            spike_wfs = np.dstack(
                [
                    data[int(spk + wf_win[0]) : int(spk + wf_win[-1]), channel_ind]
                    * channel_bit_volts
                    for spk in spikes
                ]
            )
            return spike_wfs
        else:  # if no spike found, return NaN of size (sample x channel x 1)
            return np.full((len(range(*wf_win)), len(channel_ind), 1), np.nan)

    def validate_file(self, file_type="ap"):
        file_path = self.root_dir / (self.root_name + f".{file_type}.bin")
        file_size = file_path.stat().st_size

        if file_type == "ap":
            meta = self.apmeta
        elif file_type == "lf":
            meta = self.lfmeta
        else:
            raise KeyError(f"Unknown file_type {file_type} - must be 'ap' or 'lf'")

        if file_size != meta.meta["fileSizeBytes"]:
            raise IOError(
                f"File size error! {file_path} may be corrupted or in transfer?"
            )

    def compress(self):
        from mtscomp import compress as mts_compress

        ap_file = self.root_dir / (self.root_name + ".ap.bin")
        lfp_file = self.root_dir / (self.root_name + ".lf.bin")

        meta_mapping = {"ap": self.apmeta, "lfp": self.lfmeta}

        compressed_files = []
        for bin_fp, band_type in zip([ap_file, lfp_file], ["ap", "lfp"]):
            if not bin_fp.exists():
                raise FileNotFoundError(
                    f'Compression error - "{bin_fp}" does not exist'
                )
            cbin_fp = bin_fp.parent / f"{bin_fp.stem}.cbin"
            ch_fp = bin_fp.parent / f"{bin_fp.stem}.ch"

            if cbin_fp.exists():
                assert ch_fp.exists()
                logger.info(f"Compressed file exists ({cbin_fp}), skipping...")
                continue

            try:
                mts_compress(
                    bin_fp,
                    cbin_fp,
                    ch_fp,
                    sample_rate=meta_mapping[band_type]["sample_rate"],
                    n_channels=meta_mapping[band_type]["num_channels"],
                    dtype=np.memmap(bin_fp).dtype,
                )
            except Exception as e:
                cbin_fp.unlink(missing_ok=True)
                ch_fp.unlink(missing_ok=True)
                raise e
            else:
                compressed_files.append((cbin_fp, ch_fp))

        return compressed_files

    def decompress(self):
        from mtscomp import decompress as mts_decompress

        ap_file = self.root_dir / (self.root_name + ".ap.bin")
        lfp_file = self.root_dir / (self.root_name + ".lf.bin")

        decompressed_files = []
        for bin_fp, band_type in zip([ap_file, lfp_file], ["ap", "lfp"]):
            if bin_fp.exists():
                logger.info(f"Decompressed file exists ({bin_fp}), skipping...")
                continue

            cbin_fp = bin_fp.parent / f"{bin_fp.stem}.cbin"
            ch_fp = bin_fp.parent / f"{bin_fp.stem}.ch"

            if not cbin_fp.exists():
                raise FileNotFoundError(
                    f'Decompression error - "{cbin_fp}" does not exist'
                )

            try:
                decomp_arr = mts_decompress(cbin_fp, ch_fp)
                decomp_arr.tofile(bin_fp)
            except Exception as e:
                bin_fp.unlink(missing_ok=True)
                raise e
            else:
                decompressed_files.append(bin_fp)

        return decompressed_files


class SpikeGLXMeta:
    def __init__(self, meta_filepath):
        """
        Some good processing references:
            https://billkarsh.github.io/SpikeGLX/Support/SpikeGLX_Datafile_Tools.zip
            https://github.com/jenniferColonell/Neuropixels_evaluation_tools/blob/master/SGLXMetaToCoords.m
        """

        self.fname = meta_filepath
        self.meta = _read_meta(meta_filepath)

        # Get probe part number
        self.probe_PN = self.meta.get("imDatPrb_pn", "3A")

        # Infer npx probe model (e.g. 1.0 (3A, 3B) or 2.0)
        probe_model = self.meta.get("imDatPrb_type", 0)
        if probe_model < 1:
            if "typeEnabled" in self.meta and self.probe_PN == "3A":
                self.probe_model = "neuropixels 1.0 - 3A"
            elif "typeImEnabled" in self.meta and self.probe_PN == "NP1010":
                self.probe_model = "neuropixels 1.0"
            else:
                self.probe_model = self.probe_PN
        elif probe_model == 1100:
            self.probe_model = "neuropixels UHD"
        elif probe_model == 21:
            self.probe_model = "neuropixels 2.0 - SS"
        elif probe_model == 24:
            self.probe_model = "neuropixels 2.0 - MS"
        else:
            self.probe_model = str(probe_model)

        # Get recording time
        self.recording_time = datetime.strptime(
            self.meta.get("fileCreateTime_original", self.meta["fileCreateTime"]),
            "%Y-%m-%dT%H:%M:%S",
        )
        self.recording_duration = self.meta.get("fileTimeSecs")

        # Get probe serial number - 'imProbeSN' for 3A and 'imDatPrb_sn' for 3B
        try:
            self.probe_SN = self.meta.get("imProbeSN", self.meta.get("imDatPrb_sn"))
        except KeyError:
            raise KeyError(
                "Probe Serial Number not found in"
                ' either "imProbeSN" or "imDatPrb_sn"'
            )

        # Parse channel info
        self.chanmap = (
            self._parse_chanmap(self.meta["~snsChanMap"])
            if "~snsChanMap" in self.meta
            else None
        )
        self.geommap = (
            self._parse_geommap(self.meta["~snsGeomMap"])
            if "~snsGeomMap" in self.meta
            else None
        )
        self.shankmap = (
            self._parse_shankmap(self.meta["~snsShankMap"])
            if "~snsShankMap" in self.meta
            else None
        )
        self.imroTbl = (
            self._parse_imrotbl(self.meta["~imroTbl"])
            if "~imroTbl" in self.meta
            else None
        )

        if self.shankmap is None and self.geommap is not None:
            self.shankmap = self._transform_geom_to_shank()

        # Channels being recorded, exclude Sync channels - basically a 1-1 mapping to shankmap
        self.recording_channels = np.arange(len(self.imroTbl["data"]))[
            self.get_recording_channels_indices(exclude_sync=True)
        ]

    @staticmethod
    def _parse_chanmap(raw):
        """
        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#channel-map
        Parse channel map header structure. Converts:

            '(x,y,z)(c0,x:y)...(cI,x:y),(sy0;x:y)'

        e.g:

            '(384,384,1)(AP0;0:0)...(AP383;383:383)(SY0;768:768)'

        into dict of form:

            {'shape': [x,y,z], 'c0': [x,y], ... }
        """

        res = {}
        for u in (i.rstrip(")").split(";") for i in raw.split("(") if i != ""):
            if (len(u)) == 1:
                res["shape"] = u[0].split(",")
            else:
                res[u[0]] = u[1].split(":")

        return res

    @staticmethod
    def _parse_shankmap(raw):
        """
        The shankmap contains details on the shank info
            for each electrode sites of the sites being recorded only

        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#shank-map
        Parse shank map header structure. Converts:

            '(x,y,z)(a:b:c:d)...(a:b:c:d)'

        e.g:

            '(1,2,480)(0:0:192:1)...(0:1:191:1)'

        into dict of form:

            {'shape': [x,y,z], 'data': [[a,b,c,d],...]}
        """
        res = {"shape": None, "data": []}

        for u in (i.rstrip(")") for i in raw.split("(") if i != ""):
            if "," in u:
                res["shape"] = [int(d) for d in u.split(",")]
            else:
                res["data"].append([int(d) for d in u.split(":")])

        return res

    @staticmethod
    def _parse_geommap(raw):
        """
        The shankmap contains details on the shank info
            for each electrode sites of the sites being recorded only
        Parsing from the `~snsGeomMap` (available with SpikeGLX 20230202-phase30 and later)

        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/Metadata_30.md
        Parse shank map header structure. Converts:

            '(x,y,z)(a:b:c:d)...(a:b:c:d)'
            a: zero-based shank #
            b: x-coordinate (um) of elecrode center
            c: z-coordinate (um) of elecrode center
            d: 0/1 `used` flag (included in spatial average or not)
        e.g:

            '(1,2,480)(0:0:192:1)...(0:1:191:1)'

        into dict of form:

            {'shape': [x,y,z], 'data': [[a,b,c,d],...]}
        """
        res = {"header": None, "data": []}

        for u in (i.rstrip(")") for i in raw.split("(") if i != ""):
            if "," in u:
                res["header"] = [d for d in u.split(",")]
            else:
                res["data"].append([int(d) for d in u.split(":")])

        return res

    @staticmethod
    def _parse_imrotbl(raw):
        """
        The imro table contains info for all electrode sites (no sync)
            for a particular electrode configuration (all 384 sites)
        Note: not all of these 384 sites are necessarily recorded

        https://github.com/billkarsh/SpikeGLX/blob/master/Markdown/UserManual.md#imro-per-channel-settings
        Parse imro tbl structure. Converts:

            '(X,Y,Z)(A B C D E)...(A B C D E)'

        e.g.:

            '(641251209,3,384)(0 1 0 500 250)...(383 0 0 500 250)'

        into dict of form:

            {'shape': (x,y,z), 'data': []}
        """
        res = {"shape": None, "data": []}

        for u in (i.rstrip(")") for i in raw.split("(") if i != ""):
            if "," in u:
                res["shape"] = [int(d) for d in u.split(",")]
            else:
                res["data"].append([int(d) for d in u.split(" ")])

        return res

    def _transform_geom_to_shank(self):
        if self.geommap is None:
            raise ValueError("Geometry Map not available")

        from . import probe_geometry

        probe_params = dict(
            zip(probe_geometry.geom_param_names, probe_geometry.M[self.probe_PN])
        )
        probe_params["probe_type"] = self.probe_PN
        elec_pos_df = probe_geometry.build_npx_probe(**probe_params)

        res = {"shape": self.geommap["header"], "data": []}
        for shank, x_coord, y_coord, is_used in self.geommap["data"]:
            # offset shank pitch
            x_coord += probe_params["shankPitch"] * shank
            matched_elec = elec_pos_df.query(
                f"x_coord=={x_coord} & y_coord=={y_coord} & shank=={shank}"
            )
            shank_col, shank_row = (
                matched_elec.shank_col.iloc[0],
                matched_elec.shank_row.iloc[0],
            )
            res["data"].append([shank, shank_col, shank_row, is_used])

        return res

    def get_recording_channels_indices(self, exclude_sync=False):
        """
        The indices of recorded channels (in chanmap)
         with respect to the channels listed in the imro table
        """
        recorded_chns_ind = [
            int(v[0])
            for k, v in self.chanmap.items()
            if k != "shape" and (not k.startswith("SY") if exclude_sync else True)
        ]
        orig_chns_ind = self.get_original_chans()
        _, _, chns_ind = np.intersect1d(
            orig_chns_ind, recorded_chns_ind, return_indices=True
        )
        return chns_ind

    def get_original_chans(self):
        """
        Because you can selectively save channels, the
        ith channel in the file isn't necessarily the ith acquired channel.
        Use this function to convert from ith stored to original index.

        Credit to https://billkarsh.github.io/SpikeGLX/Support/SpikeGLX_Datafile_Tools.zip
            OriginalChans() function
        """
        if self.meta["snsSaveChanSubset"] == "all":
            # output = int32, 0 to nSavedChans - 1
            channels = np.arange(0, int(self.meta["nSavedChans"]))
        else:
            # parse the channel list self.meta['snsSaveChanSubset']
            channels = np.arange(0)  # empty array
            for channel_range in self.meta["snsSaveChanSubset"].split(","):
                # a block of contiguous channels specified as chan or chan1:chan2 inclusive
                ix = [int(r) for r in channel_range.split(":")]
                assert len(ix) in (
                    1,
                    2,
                ), f"Invalid channel range spec '{channel_range}'"
                channels = np.append(channels, np.r_[ix[0] : ix[-1] + 1])
        return channels


# ============= HELPER FUNCTIONS =============


def _read_meta(meta_filepath):
    """
    Read metadata in 'k = v' format.

    The fields '~snsChanMap' and '~snsShankMap' are further parsed into
    'snsChanMap' and 'snsShankMap' dictionaries via calls to
    SpikeGLX._parse_chanmap and SpikeGLX._parse_shankmap.
    """

    res = {}
    with open(meta_filepath) as f:
        for line in (line.rstrip() for line in f):
            if "=" in line:
                try:
                    k, v = line.split("=")
                    v = convert_to_number(v)
                    res[k] = v
                except ValueError:
                    pass
    return res


def retrieve_recording_duration(meta_filepath):
    root_dir = pathlib.Path(meta_filepath).parent
    spike_glx = SpikeGLX(root_dir)
    return (
        spike_glx.apmeta.recording_duration
        or spike_glx.ap_timeseries.shape[0] / spike_glx.apmeta.meta["imSampRate"]
    )
