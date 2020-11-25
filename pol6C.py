from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *
import numpy as np

from gc import get_referents
from gc import collect
import sys
import pickle
import multiprocessing as mp
from scipy.optimize import differential_evolution
from obspy.signal.tf_misfit import cwt
from obspy.signal.util import next_pow_2
from scipy.signal import spectrogram, hanning, convolve, hilbert
from scipy.linalg import det, toeplitz
from obspy import Stream, Trace
from datetime import datetime
import matplotlib.pyplot as plt
import tables as tb
from matplotlib import colors
from scipy.interpolate import interp2d
from tqdm import tqdm


class RotPol:
    """
    Six-component polarization analysis interface
        Input parameters:
            tra_N: North component of translational motion (either an ObsPy stream object or a 1D array)

            tra_E: East component of translational motion

            tra_Z: Vertical component of translational motion

            rot_N: North component of rotational motion

            rot_E: East component of rotational motion

            rot_Z: Vertical component of rotational motion

            delta: Sampling rate (in s). Needs to be specified if the input components are not ObsPy Stream objects

            domain: Specifies if the analysis is performed in the time domain or on spectrograms
                't': Analysis is performed in the time-domain
                'f': Analysis is performed on a time-frequency representation of the input signal
            method: Specifies the way that the polarization fit is determined
                'MUSIC': MUSIC algorithm
                'ML': Maximum likelihood method
                'DOT': Minimize great-circle distance / dot  product (corresponding to the angle between the model
                       polarization vector and the polarization direction in the data) on the 5-sphere between the
                       measured and tested polarization vector (DEFAULT)

            search: Specifies the optimization procedure
                'grid': Simple grid search across the parameter space (DEFAULT)
                'global': Global optimization for speed-up using a differential evolution algorithm

            Range of parameters to be tested. List like objects of the form range=[range_min, range_max, range_increment]
            If 'search' == 'global', the range_increment argument is ignored
                vl_range: Range of Love wave velocities (in m/s) to be tested.
                vr_range: Range of Rayleigh wave velocities (in m/s) to be tested.
                vp_range: Range of P-wave velocities (in m/s) to be tested.
                vs_range: Range of S-wave velocities (in m/s) to be tested.
                theta_range: Range of incidence angles (in degrees) to be tested.
                phi_range: Range of azimuth angles (in degrees) to be tested.
                xi_range: Range of Rayleigh wave ellipticity angles (in rad) to be tested.

            free_surface: True (Default) or False. Specifies whether the recording station is located at the
                          free surface

            v_scal: Scaling velocity (in m/s) to ensure numerical stability. Ideally, v_scal is close to the S-Wave
                    velocity at the receiver

            window_length: Window length (in s) of the sliding analysis window for the time-domain analysis.

            overlap: Between 0 and 1. Percentage of overlap between neighbouring analysis windows (DEFAULT=0.5).




      Methods:
          estimate_p():     Estimates P-wave parameters.
          estimate_sv():    Estimates SV-wave parameters.
          estimate_sh():    Estimates SH-wave parameters.
          estimate_l():     Estimates Love wave parameters.
          estimate_r():     Estimates Rayleigh wave parameters.
          estimate_all():   Sequentially estimates parameters for all wave types.


      """

    def __init__(self, traN=None, traE=None, traZ=None, rotN=None, rotE=None, rotZ=None, delta=None, domain='t',
                 method='DOT', spectrogram='st', search='grid', music_nullspace='auto',
                 vl_range=None,
                 vr_range=None, vp_range=None, vs_range=None, theta_range=None, phi_range=None, xi_range=None, v_scal=1,
                 spect=None, window=None, free_surface=True, dsfact=1, dsfacf=1, dop_min=0, frange=None):
        self.spectrogram = spectrogram
        if spect is None:
            self.spect = \
                {'kind': spectrogram, 'window_length': 10., 'overlap': 0.5, 'fmin': 0.005,
                 'fmax': 1,
                 'nf': 100, 'w0': 10}
        else:
            self.spect = spect
        self.dop_min = dop_min
        self.window_length = self.spect['window_length']
        self.overlap = self.spect['overlap']
        self.frange = frange
        if window is None:
            self.window = {'window_length_periods': 1., 'window_length_frequencies': 0.01}
        else:
            self.window = window

        self.traN, self.traE, self.traZ, self.rotN, self.rotE, self.rotZ = traN, traE, traZ, rotN, rotE, rotZ
        self.delta = delta
        self.xi_range = xi_range
        self.phi_range = phi_range
        self.theta_range = theta_range
        self.vs_range = vs_range
        self.vp_range = vp_range
        self.vr_range = vr_range
        self.vl_range = vl_range
        self.domain = domain
        self.method = method
        self.search = search
        self.free_surface = free_surface
        self.music_nullspace = music_nullspace
        self.v_scal = v_scal
        self.dsfacf = dsfacf
        self.dsfact = dsfact
        self.computed = {"P": False, "SV": False, "SH": False, "L": False, "R": False}
        self.wave_parameters = {"P": [], "SV": [], "SH": [], "L": [], "R": []}
        if isinstance(self.traN, Stream):
            assert isinstance(self.traE, Stream) and isinstance(self.traZ, Stream) and isinstance(self.rotN, Stream) \
                   and isinstance(self.rotE, Stream) and isinstance(self.rotZ, Stream), \
                "All components must be of the same type! Either an ObsPy Stream object or a NumPy array."
            self.delta = self.traN[0].stats.delta
        else:
            if self.delta is None:
                sys.exit("The sampling rate 'delta' needs to be specified!")
            self.traN = Stream(Trace(traN,
                                     header={"delta": delta, "npts": int(len(traN)),
                                             "sampling_rate": float(1 / delta),
                                             "channel": "TLN", "starttime": 0.}))
            self.traE = Stream(Trace(traE,
                                     header={"delta": delta, "npts": int(len(traE)),
                                             "sampling_rate": float(1 / delta),
                                             "channel": "TLE", "starttime": 0.}))
            self.traZ = Stream(Trace(traZ,
                                     header={"delta": delta, "npts": int(len(traZ)),
                                             "sampling_rate": float(1 / delta),
                                             "channel": "TLZ", "starttime": 0.}))
            self.rotN = Stream(Trace(rotN,
                                     header={"delta": delta, "npts": int(len(rotN)),
                                             "sampling_rate": float(1 / delta),
                                             "channel": "RTN", "starttime": 0.}))
            self.rotE = Stream(Trace(rotE,
                                     header={"delta": delta, "npts": int(len(rotE)),
                                             "sampling_rate": float(1 / delta),
                                             "channel": "RTE", "starttime": 0.}))
            self.rotZ = Stream(Trace(rotZ,
                                     header={"delta": delta, "npts": int(len(rotZ)),
                                             "sampling_rate": float(1 / delta),
                                             "channel": "RTZ", "starttime": 0.}))

        self.time = self.traN[0].times(type="utcdatetime")
        self.window_n = 2 * int((self.spect['window_length'] / self.delta) / 2)  # Window length in samples, even number

        if self.spect["kind"] == 'spec' or self.spect["kind"] == 'cwt':
            nfft = next_pow_2(self.window_n) * 2
            nfft = np.max([nfft, 2 * 256])

        else:
            nfft = (self.traN[0].stats.npts)

        if self.frange is None:
            self.frange = [0, 1 / self.delta / 2]

        h5_file = tb.open_file('tmp.hdf', 'w')

        if self.domain == "f":

            for traN, traE, traZ, rotN, rotE, rotZ in zip(self.traN, self.traE, self.traZ, self.rotN, self.rotE,
                                                          self.rotZ):
                if traN.stats.npts < self.window_n * 4:
                    continue
                print("Performing time-frequency decomposition...\n ")
                self.f_pol, self.t_pol, u1, u2, u3, u4, u5, u6 = \
                    _compute_spec(traN, traE, traZ, rotN, rotE, rotZ, kind=self.spect["kind"], window_n=self.window_n,
                                  nfft=nfft, overlap=self.spect["overlap"], fmin=self.spect["fmin"],
                                  fmax=self.spect["fmax"], nf=self.spect["nf"], w0=self.spect["w0"], dsfacf=self.dsfacf)

                nfsum, ntsum, dsfacf, dsfact = _calc_dop_windows(
                    self.window["window_length_frequencies"], self.window["window_length_periods"], self.delta,
                    self.spect["fmax"],
                    self.spect["fmin"], self.spect["kind"], self.spect["nf"], nfft, self.spect["overlap"],
                    self.spect["window_length"], np.float(self.time[-1] - self.time[0]), self.f_pol)
                ntsum[ntsum > u1.shape[1]] = u1.shape[1]
                dsfact = self.dsfact

                print("Setting up spectral matrices and saving them to a temporary HDF5 file...\n ")
                u = h5_file.create_carray(h5_file.root, 'u', obj=np.moveaxis(np.array([u1, u2, u3, u4, u5, u6]), 0, 2))
                collect()
                S = h5_file.create_carray(h5_file.root, 'S', obj=np.einsum('...i,...j->...ij', np.conj(u), u))
                collect()
                if self.spect["kind"] == 'spec':
                    # 2D Hann window to smooth the covariance matrices
                    w = np.einsum('...i,...j->...ij', hanning(nfsum + 2)[1:-1], hanning(ntsum + 2)[1:-1])
                    w /= np.sum(w)
                    print("Smoothing spectral matrices...\n ")
                    for j in range(S.shape[2]):
                        for k in range(S.shape[3]):
                            S[:, :, j, k] = \
                                convolve(S[..., j, k].real, w, mode='same') + \
                                convolve(S[..., j, k].imag, w, mode='same') * 1j
                elif self.spect["kind"] == 'cwt' or self.spect["kind"] == 'st':
                    print("Smoothing spectral matrices...\n ")
                    w_f = hanning(nfsum + 2)[1:-1] * np.ones((nfsum, 1))
                    w_f /= np.sum(w_f, axis=0)

                    for j in range(S.shape[2]):
                        for k in range(S.shape[3]):
                            S[:, :, j, k] = convolve(S[..., j, k].real, w_f, mode='same') + \
                                            convolve(S[..., j, k].imag, w_f, mode='same') * 1j
                            for i in range(S.shape[0]):
                                if ntsum[i] >= 3:
                                    w_t = hanning(ntsum[i])
                                    w_t /= np.sum(w_t)
                                    S[i, :, j, k] = \
                                        convolve(S[i, :, j, k].real, w_t, mode='same') + \
                                        convolve(S[i, :, j, k].imag, w_t, mode='same') * 1j

                print("Down-sampling spectral matrix...\n")
                h5_file.rename_node(h5_file.root, 'S_full', name='S', overwrite=True)
                h5_file.create_carray(h5_file.root, 'S', obj=h5_file.root.S_full[:, ::dsfact, :, :])
                idx1 = (np.abs(self.f_pol - self.frange[0])).argmin()
                idx2 = (np.abs(self.f_pol - self.frange[1])).argmin()
                self.f_pol = self.f_pol[idx1:idx2 + 1]
                h5_file.rename_node(h5_file.root, 'S_full', name='S', overwrite=True)
                h5_file.create_carray(h5_file.root, 'S', obj=h5_file.root.S_full[idx1:idx2 + 1, :, :, :])
                h5_file.remove_node(h5_file.root, 'S_full')

                self.t_pol = self.time[::dsfact]

        elif self.domain == 't':
            start, stop, incr = int(self.window_n / 2), -1 - int(self.window_n / 2), \
                                np.max([1, int((1 - self.overlap) * self.window_n)])
            self.t_pol = self.time[start:stop:incr]
            self.f_pol = np.empty(1)
            u1, u2, u3, u4, u5, u6 = hilbert(traN), hilbert(traE), hilbert(traZ), \
                                     hilbert(rotN), hilbert(rotE), hilbert(rotZ)
            u = np.array([u1, u2, u3, u4, u5, u6])
            u.shape = (u.shape[0], u.shape[1], 1)
            u = np.moveaxis(u, 0, 2)
            u = np.moveaxis(u, 1, 0)
            u = h5_file.create_carray(h5_file.root, 'u', obj=u)
            h5_file.create_carray(h5_file.root, 'S', obj=np.einsum('...i,...j->...ij', np.conj(u), u))
        h5_file.close()
        print("Ready to compute wave parameters!\n")

    def estimate_all(self):
        self.estimate_p()
        self.estimate_sv()
        self.estimate_sh()
        self.estimate_l()
        self.estimate_r()

    def estimate_p(self):
        h5_file = tb.open_file('tmp.hdf', 'r')
        S = h5_file.root.S
        if self.computed['P']:
            print("P wave polarization attributes are already computed!")
            return
        assert self.vp_range and self.vs_range and self.theta_range and self.phi_range, \
            "For P-wave analysis, vp_range, vs_range, theta_range, and phi_range need to be specified!"

        ss = SearchSpace(wave_type='P', vp=self.vp_range, vs=self.vs_range, theta=self.theta_range,
                         phi=self.phi_range, v_scal=self.v_scal, free_surface=self.free_surface)
        if self.domain == 'f':

            for j in tqdm(range(S.shape[0] * S.shape[1]), desc='Estimating P-wave parameters'):
                loc = np.unravel_index(j, (S.shape[0], S.shape[1]))
                est = Estimator(ss, C=S[loc[0], loc[1], :, :], method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["P"].append(est.solve())
        else:
            start, incr = int(self.window_n / 2), np.max([1, int((1 - self.overlap) * self.window_n)])
            w = hanning(self.window_n + 2)[1:-1]
            w /= sum(w)
            for j in tqdm(range(len(self.t_pol)), desc='Estimating P-wave parameters'):
                C = np.zeros((6, 6)) + 1j * np.zeros((6, 6))
                for o in range(S.shape[2]):
                    for p in range(S.shape[3]):
                        C[o, p] = np.sum(w * S[0, start + j * incr - int(self.window_n / 2):start + j * incr + int(
                            self.window_n / 2), o, p])
                est = Estimator(ss, C=C, method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["P"].append(est.solve())
        self.computed["P"] = True
        h5_file.close()

    def estimate_sv(self):
        h5_file = tb.open_file('tmp.hdf', 'r')
        S = h5_file.root.S
        if self.computed['SV']:
            print("SV wave polarization attributes are already computed!")
            return
        assert self.vp_range and self.vs_range and self.theta_range and self.phi_range, \
            "For SV-wave analysis, vp_range, vs_range, theta_range, and phi_range need to be specified!"

        ss = SearchSpace(wave_type='SV', vp=self.vp_range, vs=self.vs_range, theta=self.theta_range,
                         phi=self.phi_range, v_scal=self.v_scal, free_surface=self.free_surface)
        if self.domain == 'f':

            for j in tqdm(range(S.shape[0] * S.shape[1]), desc='Estimating SV-wave parameters'):
                loc = np.unravel_index(j, (S.shape[0], S.shape[1]))
                est = Estimator(ss, C=S[loc[0], loc[1], :, :], method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["SV"].append(est.solve())
        else:
            start, incr = int(self.window_n / 2), np.max([1, int((1 - self.overlap) * self.window_n)])
            w = hanning(self.window_n + 2)[1:-1]
            w /= sum(w)
            for j in tqdm(range(len(self.t_pol)), desc="Estimating SV-wave parameters"):
                C = np.zeros((6, 6)) + 1j * np.zeros((6, 6))
                for o in range(S.shape[2]):
                    for p in range(S.shape[3]):
                        C[o, p] = np.sum(w * S[0, start + j * incr - int(self.window_n / 2):start + j * incr + int(
                            self.window_n / 2), o, p])
                est = Estimator(ss, C=C, method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["SV"].append(est.solve())
        self.computed["SV"] = True
        h5_file.close()

    def estimate_sh(self):
        h5_file = tb.open_file('tmp.hdf', 'r')
        S = h5_file.root.S
        if self.computed['SH']:
            print("SH wave polarization attributes are already computed!")
            return
        assert self.vs_range and self.theta_range and self.phi_range, \
            "For SH-wave analysis, vs_range, theta_range, and phi_range need to be specified!"

        ss = SearchSpace(wave_type='SH', vs=self.vs_range, theta=self.theta_range,
                         phi=self.phi_range, v_scal=self.v_scal, free_surface=self.free_surface)
        if self.domain == 'f':

            for j in tqdm(range(S.shape[0] * S.shape[1]), desc='Estimating SH-wave parameters'):
                loc = np.unravel_index(j, (S.shape[0], S.shape[1]))
                est = Estimator(ss, C=S[loc[0], loc[1], :, :], method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["SH"].append(est.solve())
        else:
            start, incr = int(self.window_n / 2), np.max([1, int((1 - self.overlap) * self.window_n)])
            w = hanning(self.window_n + 2)[1:-1]
            w /= sum(w)
            for j in tqdm(range(len(self.t_pol)), desc='Estimating SH-wave parameters'):
                C = np.zeros((6, 6)) + 1j * np.zeros((6, 6))
                for o in range(S.shape[2]):
                    for p in range(S.shape[3]):
                        C[o, p] = np.sum(w * S[0, start + j * incr - int(self.window_n / 2):start + j * incr + int(
                            self.window_n / 2), o, p])
                est = Estimator(ss, C=C, method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["SH"].append(est.solve())
        self.computed["SH"] = True
        h5_file.close()

    def estimate_r(self):
        h5_file = tb.open_file('tmp.hdf', 'r')
        S = h5_file.root.S
        if self.computed['R']:
            print("Rayleigh wave polarization attributes are already computed!")
            return
        assert self.vr_range and self.phi_range and self.xi_range, \
            "For Rayleigh-wave analysis, vr_range, phi_range, and xi_range need to be specified!"

        ss = SearchSpace(wave_type='R', vr=self.vr_range, xi=self.xi_range,
                         phi=self.phi_range, v_scal=self.v_scal, free_surface=self.free_surface)
        if self.domain == 'f':

            for j in tqdm(range(S.shape[0] * S.shape[1]), desc='Estimating Rayleigh wave parameters'):
                loc = np.unravel_index(j, (S.shape[0], S.shape[1]))
                est = Estimator(ss, C=S[loc[0], loc[1], :, :], method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["R"].append(est.solve())
        else:
            start, incr = int(self.window_n / 2), np.max([1, int((1 - self.overlap) * self.window_n)])
            w = hanning(self.window_n + 2)[1:-1]
            w /= sum(w)
            for j in tqdm(range(len(self.t_pol)), desc='Estimating Rayleigh wave parameters'):
                C = np.zeros((6, 6)) + 1j * np.zeros((6, 6))
                for o in range(S.shape[2]):
                    for p in range(S.shape[3]):
                        C[o, p] = np.sum(w * S[0, start + j * incr - int(self.window_n / 2):start + j * incr + int(
                            self.window_n / 2), o, p])
                est = Estimator(ss, C=C, method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["R"].append(est.solve())
        self.computed["R"] = True
        h5_file.close()

    def estimate_l(self):
        h5_file = tb.open_file('tmp.hdf', 'r')
        S = h5_file.root.S
        if self.computed['L']:
            print("Love wave polarization attributes are already computed!")
            return
        assert self.vl_range and self.phi_range, \
            "For Love-wave analysis, vl_range, and phi_range need to be specified!"

        ss = SearchSpace(wave_type='L', vl=self.vl_range,
                         phi=self.phi_range, v_scal=self.v_scal, free_surface=self.free_surface)
        if self.domain == 'f':
            for j in tqdm(range(S.shape[0] * S.shape[1]), desc='Estimating Love wave parameters'):
                loc = np.unravel_index(j, (S.shape[0], S.shape[1]))
                est = Estimator(ss, C=S[loc[0], loc[1], :, :], method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["L"].append(est.solve())
        else:
            start, incr = int(self.window_n / 2), np.max([1, int((1 - self.overlap) * self.window_n)])
            w = hanning(self.window_n + 2)[1:-1]
            w /= sum(w)
            for j in tqdm(range(len(self.t_pol)), desc='Estimating Love wave parameters'):
                C = np.zeros((6, 6)) + 1j * np.zeros((6, 6))
                for o in range(S.shape[2]):
                    for p in range(S.shape[3]):
                        C[o, p] = np.sum(w * S[0, start + j * incr - int(self.window_n / 2):start + j * incr + int(
                            self.window_n / 2), o, p])
                est = Estimator(ss, C=C, method=self.method,
                                music_nullspace=self.music_nullspace,
                                search=self.search, dop_min=self.dop_min)
                self.wave_parameters["L"].append(est.solve())
        self.computed["L"] = True
        h5_file.close()

    def save(self, name=None):
        if name is None:
            name = "PolRot." + str(datetime.now().strftime("%d-%m-%Y.%H.%M.%S")) + ".pkl"
        elif not isinstance(name, str):
            raise ValueError("Name must be a string!")

        fid = open(name, 'wb')
        pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)
        fid.close()

    def get_theta(self, wave_type):
        wtype_list = ["P", "SV", "SH", "L", "R"]
        if wave_type not in wtype_list:
            raise ValueError(f"{wave_type} is not a valid wave type!")

        if wave_type == 'L' or wave_type == 'R':
            raise ValueError(f"No parameter 'theta' for wave type: '{wave_type}'!")

        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        theta = np.array([o.theta for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            theta = np.reshape(theta, (len(self.f_pol), len(self.t_pol)))
        return theta

    def get_phi(self, wave_type):
        wtype_list = ["P", "SV", "SH", "L", "R"]
        if wave_type not in wtype_list:
            raise ValueError(f"{wave_type} is not a valid wave type!")

        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        phi = np.array([o.phi for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            phi = np.reshape(phi, (len(self.f_pol), len(self.t_pol)))
        return phi

    def get_vp(self, wave_type):
        wtype_list = ["P", "SV", "SH", "L", "R"]
        if wave_type not in wtype_list:
            raise ValueError(f"{wave_type} is not a valid wave type!")

        if wave_type == 'L' or wave_type == 'R' or wave_type == 'SH':
            raise ValueError(f"No parameter 'vp' for wave type: '{wave_type}'!")

        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        vp = np.array([o.vp for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            vp = np.reshape(vp, (len(self.f_pol), len(self.t_pol)))
        return vp

    def get_vs(self, wave_type):
        wtype_list = ["P", "SV", "SH", "L", "R"]
        if wave_type not in wtype_list:
            raise ValueError(f"{wave_type} is not a valid wave type!")

        if wave_type == 'L' or wave_type == 'R':
            raise ValueError(f"No parameter 'vs' for wave type: '{wave_type}'!")

        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        vs = np.array([o.vs for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            vs = np.reshape(vs, (len(self.f_pol), len(self.t_pol)))
        return vs

    def get_vl(self):
        wave_type = "L"
        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        vl = np.array([o.vl for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            vl = np.reshape(vl, (len(self.f_pol), len(self.t_pol)))
        return vl

    def get_vr(self):
        wave_type = "R"
        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        vr = np.array([o.vr for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            vr = np.reshape(vr, (len(self.f_pol), len(self.t_pol)))
        return vr

    def get_xi(self):
        wave_type = "R"
        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        xi = np.array([o.xi for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            xi = np.reshape(xi, (len(self.f_pol), len(self.t_pol)))
        return xi

    def get_lh(self, wave_type):
        wtype_list = ["P", "SV", "SH", "L", "R"]
        if wave_type not in wtype_list:
            raise ValueError(f"{wave_type} is not a valid wave type!")

        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        lh = np.array([o.Lmax for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            lh = np.reshape(lh, (len(self.f_pol), len(self.t_pol)))
        return lh

    def get_dop(self, wave_type):
        wtype_list = ["P", "SV", "SH", "L", "R"]
        if wave_type not in wtype_list:
            raise ValueError(f"{wave_type} is not a valid wave type!")

        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters for wave type {wave_type} are not yet computed!")

        dop = np.array([o.dop for o in self.wave_parameters[wave_type]])
        if self.domain == 'f':
            dop = np.reshape(dop, (len(self.f_pol), len(self.t_pol)))
        return dop

    def separate(self, wave_type, vmin=0.6, vmax=0.85):
        wtype_list = ["P", "SV", "SH", "L", "R"]

        if wave_type not in wtype_list:
            raise ValueError(f"{wave_type} is not a valid wave type! Must either be 'P', 'SV', 'SH', 'L', or 'R'")

        alphas = self.get_lh(wave_type)
        alphas = colors.Normalize(vmin=vmin, vmax=vmax)(alphas)
        alphas[alphas < 0.] = 0.
        alphas[alphas > 1.] = 1.
        interpolator = interp2d(np.asarray(self.t_pol-self.t_pol[0], dtype='float'), self.f_pol, alphas)
        trace_list = [self.traN, self.traE, self.traZ, self.rotN, self.rotE, self.rotZ]
        trace_sep_list = []

        for tr in trace_list:
            tr_stran, f_stran = s_transform(tr[0].data)
            idx1 = (np.abs(f_stran - self.frange[0])).argmin()
            idx2 = (np.abs(f_stran - self.frange[1])).argmin()
            alpha_int = np.zeros((f_stran.shape[0], self.time.shape[0]))
            alpha_int[idx1:idx2+1, :] = interpolator(np.asarray(self.time-self.time[0], dtype='float'), f_stran[idx1:idx2+1])
            tr_sep = np.multiply(alpha_int, tr_stran)  # Filter spectrograms with wave-type likelihood
            tr_sep = np.fft.irfft(np.sum(tr_sep, axis=1))  # Inverse S-tranform
            trace_sep_list.append(tr_sep)

        return trace_sep_list

    def plot_wave_parameters(self, wave_type):
        wtype_list = ["P", "SV", "SH", "L", "R", "all"]
        if wave_type not in wtype_list:
            raise ValueError(f"Invalid wave type: {wave_type}")

        if not self.computed[wave_type]:
            raise ValueError(f"Wave parameters are not yet computed for wave type: {wave_type}!")

        if wave_type == "P":
            vp = self.get_vp(wave_type)
            vs = self.get_vs(wave_type)
            theta = self.get_theta(wave_type)
            phi = self.get_phi(wave_type)
            fig, ax = plt.subplots(5, 1, sharex=True)

            if vp.ndim == 1:
                ax[9] = plt.plot(self.time, self.traN[9].data)
                plt.ylabel("Amplitude")
                ax[1] = plt.plot(self.t_pol, vp, '.', color='k')
                plt.ylabel("P-wave velocity (m/s)")
                ax[2] = plt.plot(self.t_pol, vs)
                plt.ylabel("S-wave velocity (m/s)")
            else:
                pass

        plt.show()

    @property
    def size(self):
        """computes the total size (in bytes) of a PolRot object & its attributes."""
        seen_ids = set()
        size = 0
        objects = [self]
        while objects:
            need_referents = []
            for obj in objects:
                if id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    size += sys.getsizeof(obj)
                    need_referents.append(obj)
            objects = get_referents(*need_referents)
        return size


class Estimator:

    def __init__(self, search_space, C=None, method='ML', cpu_count=mp.cpu_count(), music_nullspace='auto',
                 music_nullspace_threshold=0.1, search='grid', dop_min=0):
        self.method = method
        self.C = C / np.linalg.norm(C)
        assert isinstance(search_space, SearchSpace)
        self.search_space = search_space
        self.wave_type = search_space.wave_type
        self.cpu_count = cpu_count
        self.search = search
        self._set_l_shape()
        self.dop_min = dop_min

        self.music_nullspace, self.music_nullspace_threshold = music_nullspace, music_nullspace_threshold
        if self.method == 'MUSIC' or self.method == 'DOT':
            self.evalues, self.evectors = np.linalg.eig(C)
            indices = np.flip(np.argsort(np.real(self.evalues)))
            self.evalues = self.evalues[indices]
            self.evectors = self.evectors[:, indices]

            u1 = self.evectors[:, 0]
            u1 = u1 / np.linalg.norm(u1)
            gamma = np.arctan2(2 * np.dot(u1.real, u1.imag),
                               np.dot(u1.real, u1.real) -
                               np.dot(u1.imag, u1.imag))
            phi = -0.5 * gamma

            self.evectors = np.exp(1j * phi) * self.evectors
            self.nspace_dim = 6 - np.argmax(
                np.abs(self.evalues) < np.abs(self.music_nullspace_threshold * np.abs(self.evalues[0])))
            self.nspace_dim = np.min([self.nspace_dim, 5])
            self.dop = ((self.evalues[0] - self.evalues[1]) ** 2
                        + (self.evalues[0] - self.evalues[2]) ** 2
                        + (self.evalues[0] - self.evalues[3]) ** 2
                        + (self.evalues[0] - self.evalues[4]) ** 2
                        + (self.evalues[0] - self.evalues[5]) ** 2
                        + (self.evalues[1] - self.evalues[2]) ** 2
                        + (self.evalues[1] - self.evalues[3]) ** 2
                        + (self.evalues[1] - self.evalues[4]) ** 2
                        + (self.evalues[1] - self.evalues[5]) ** 2
                        + (self.evalues[2] - self.evalues[3]) ** 2
                        + (self.evalues[2] - self.evalues[4]) ** 2
                        + (self.evalues[2] - self.evalues[5]) ** 2
                        + (self.evalues[3] - self.evalues[4]) ** 2
                        + (self.evalues[3] - self.evalues[5]) ** 2
                        + (self.evalues[4] - self.evalues[5]) ** 2) / (5 * np.sum(self.evalues) ** 2)

    def solve(self):

        if self.search == 'grid':
            if self.dop >= self.dop_min:
                p = mp.Pool(self.cpu_count)
                indices = np.arange(0, self.search_space.N)
                if self.method == 'ML':
                    L = np.asarray(p.map(self._evaluate_ml, indices))
                elif self.method == 'LP':
                    L = np.asarray(p.map(self._evaluate_lp, indices))
                elif self.method == 'MUSIC':
                    L = np.asarray(p.map(self._evaluate_music, indices))
                    L = np.log(L)
                elif self.method == 'DOT':
                    L = np.asarray(p.map(self._evaluate_dist, indices))
                else:
                    L = None
                p.close()
                Lmax = np.exp(np.nanmax(np.real(L)))
                index = np.nanargmax(L)
                loc = np.unravel_index(index, self.l_shape, 'C')
                wave_estimated = self._estimated_wave_grid(loc)
            else:
                Lmax = 0.
                wave_estimated = EstimatedWave(wave_type=self.wave_type, isnone=True)
            # wave_estimated.L = abs(np.exp(L))

        elif self.search == 'global':
            if self.dop >= self.dop_min:

                bounds = self._bounds

                if self.method == 'ML':
                    result = differential_evolution(self._evaluate_ml_de, bounds, updating='deferred',
                                                    workers=self.cpu_count, disp=False)
                elif self.method == 'LP':
                    result = differential_evolution(self._evaluate_lp_de, bounds, updating='deferred',
                                                    workers=self.cpu_count, disp=False)
                elif self.method == 'MUSIC':
                    result = differential_evolution(self._evaluate_music_de, bounds, updating='deferred',
                                                    workers=self.cpu_count, disp=False)
                    result.fun = -np.log(-result.fun)
                elif self.method == 'DOT':
                    result = differential_evolution(self._evaluate_dist_de, bounds, updating='deferred',
                                                    workers=self.cpu_count, disp=False)
                else:
                    result = None

                Lmax = float(np.exp(-result.fun))
                x = result.x
                wave_estimated = self._estimated_wave_global(x)

            else:
                Lmax = 0.
                wave_estimated = EstimatedWave(wave_type=self.wave_type, isnone=True)
        else:
            sys.exit("No valid search method specified in Estimator object! Set to 'grid' or 'global'")

        wave_estimated.Lmax = abs(Lmax)
        wave_estimated.dop = self.dop
        return wave_estimated

    def _evaluate_ml(self, indices):
        location = np.unravel_index(indices, self.l_shape, 'C')
        test_wave = self._test_wave_grid(location)
        h = np.asarray(test_wave.polarization, dtype=complex)
        return -np.log(np.matmul(np.matmul(h.conj().T, np.linalg.inv(self.C)), h))

    def _evaluate_lp(self, indices):
        location = np.unravel_index(indices, self.l_shape, 'C')
        test_wave = self._test_wave_grid(location)
        h = np.asarray(test_wave.polarization, dtype=complex)
        L = np.real(np.matmul(np.matmul(h.conj().T, self.C), h))
        if L < 0:
            L = 1e-11
        return np.log(L)

    def _evaluate_music(self, indices):
        location = np.unravel_index(indices, self.l_shape, 'C')
        test_wave = self._test_wave_grid(location)
        h = np.asarray(test_wave.polarization, dtype=complex)
        Q = (self.evectors[:, 6 - self.nspace_dim:6]).dot(np.matrix.getH(self.evectors[:, 6 - self.nspace_dim:6]))
        return np.real(1. / (np.matmul(np.matmul(h.conj().T, Q), h) + 1e-11)) / 1e11

    def _evaluate_dist(self, indices):
        location = np.unravel_index(indices, self.l_shape, 'C')
        test_wave = self._test_wave_grid(location)
        h = test_wave.polarization
        return -np.float(np.abs(np.arccos(np.around(np.abs(np.real(np.dot(h.H, self.evectors[:, 0]))), 12)))) ** 2

    def _evaluate_ml_de(self, x):
        test_wave = self._test_wave_global(x)
        h = np.asarray(test_wave.polarization, dtype=complex)
        L = np.real(np.matmul(np.matmul(h.conj().T, self.C), h))
        if L < 0:
            L = 1e-11
        return np.log(np.matmul(np.matmul(h.conj().T, np.linalg.inv(self.C)), h))

    def _evaluate_lp_de(self, x):
        test_wave = self._test_wave_global(x)
        h = np.asarray(test_wave.polarization, dtype=complex)
        L = np.real(np.matmul(np.matmul(h.conj().T, self.C), h))
        if L < 0:
            L = 1e-11
        return np.nan_to_num(-np.log(L), nan=3)

    def _evaluate_music_de(self, x):
        test_wave = self._test_wave_global(x)
        h = np.asarray(test_wave.polarization, dtype=complex)
        Q = (self.evectors[:, 6 - self.nspace_dim:6]).dot(np.matrix.getH(self.evectors[:, 6 - self.nspace_dim:6]))
        L = -np.abs(1. / (np.matmul(np.matmul(h.conj().T, Q), h) + 1e-11)) / 1e11

        return np.nan_to_num(np.real(L), nan=1)

    def _evaluate_dist_de(self, x):
        test_wave = self._test_wave_global(x)
        h = test_wave.polarization
        L = np.float(np.abs(np.arccos(np.around(np.abs(np.real(np.dot(h.H, self.evectors[:, 0]))), 12)))) ** 2

        return np.nan_to_num(L, nan=np.pi)

    def _set_l_shape(self):
        if (self.wave_type == 'P' or self.wave_type == 'SV') and self.search_space.free_surface:
            self.l_shape = np.array([self.search_space.n_vp, self.search_space.n_vs,
                                     self.search_space.n_theta, self.search_space.n_phi])
        elif self.wave_type == 'P' and not self.search_space.free_surface:
            self.l_shape = np.array([self.search_space.n_theta, self.search_space.n_phi])

        elif (self.wave_type == 'SV' or self.wave_type == 'SH') and not self.search_space.free_surface:
            self.l_shape = np.array([self.search_space.n_vs, self.search_space.n_theta, self.search_space.n_phi])

        elif self.wave_type == 'SH' and self.search_space.free_surface:
            self.l_shape = np.array([self.search_space.n_vs, self.search_space.n_theta, self.search_space.n_phi])

        elif self.wave_type == 'R':
            self.l_shape = np.array([self.search_space.n_vr, self.search_space.n_phi, self.search_space.n_xi])

        elif self.wave_type == 'L':
            self.l_shape = np.array([self.search_space.n_vl, self.search_space.n_phi])

    def _estimated_wave_grid(self, loc):
        if (self.wave_type == 'P' or self.wave_type == 'SV') and self.search_space.free_surface:
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vp=self.search_space.vp_vect[loc[0]],
                                           vs=self.search_space.vs_vect[loc[1]],
                                           theta=self.search_space.theta_vect[loc[2]],
                                           phi=self.search_space.phi_vect[loc[3]])
        elif self.wave_type == 'P' and not self.search_space.free_surface:
            wave_estimated = EstimatedWave(wave_type=self.wave_type, theta=self.search_space.theta_vect[loc[0]],
                                           phi=self.search_space.phi_vect[loc[1]])
        elif (self.wave_type == 'SV' or self.wave_type == 'SH') and not self.search_space.free_surface:
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vs=self.search_space.vs_vect[loc[0]],
                                           theta=self.search_space.theta_vect[loc[1]],
                                           phi=self.search_space.phi_vect[loc[2]])
        elif self.wave_type == 'SH' and self.search_space.free_surface:
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vs=self.search_space.vs_vect[loc[0]],
                                           theta=self.search_space.theta_vect[loc[1]],
                                           phi=self.search_space.phi_vect[loc[2]])
        elif self.wave_type == 'R':
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vr=self.search_space.vr_vect[loc[0]],
                                           phi=self.search_space.phi_vect[loc[1]],
                                           xi=self.search_space.xi_vect[loc[2]])
        elif self.wave_type == 'L':
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vl=self.search_space.vl_vect[loc[0]],
                                           phi=self.search_space.phi_vect[loc[1]])
        else:
            sys.exit('Invalid wave type specified in Estimator object!')
        return wave_estimated

    def _estimated_wave_global(self, x):
        if (self.wave_type == 'P' or self.wave_type == 'SV') and self.search_space.free_surface:
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vp=x[0],
                                           vs=x[1], theta=x[2],
                                           phi=x[3])
        elif self.wave_type == 'P' and not self.search_space.free_surface:
            wave_estimated = EstimatedWave(wave_type=self.wave_type, theta=x[0],
                                           phi=x[1], free_surface=False)
        elif (self.wave_type == 'SV' or self.wave_type == 'SH') and not self.search_space.free_surface:
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vs=x[0],
                                           theta=x[1],
                                           phi=x[2], free_surface=False)
        elif self.wave_type == 'SH' and self.search_space.free_surface:
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vs=x[0],
                                           theta=x[1],
                                           phi=x[2])
        elif self.wave_type == 'R':
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vr=x[0],
                                           phi=x[1],
                                           xi=x[2])
        elif self.wave_type == 'L':
            wave_estimated = EstimatedWave(wave_type=self.wave_type, vl=x[0],
                                           phi=x[1])
        else:
            sys.exit('Invalid wave type specified in Estimator object!')
        return wave_estimated

    @property
    def _bounds(self):
        if (self.wave_type == 'P' or self.wave_type == 'SV') and self.search_space.free_surface:
            bounds = [(self.search_space.vp[0], self.search_space.vp[1]),
                      (self.search_space.vs[0], self.search_space.vs[1]),
                      (self.search_space.theta[0], self.search_space.theta[1]),
                      (self.search_space.phi[0], self.search_space.phi[1])]
        elif self.wave_type == 'P' and not self.search_space.free_surface:
            bounds = [(self.search_space.theta[0], self.search_space.theta[1]),
                      (self.search_space.phi[0], self.search_space.phi[1])]
        elif (self.wave_type == 'SV' or self.wave_type == 'SH') and not self.search_space.free_surface:
            bounds = [(self.search_space.vs[0], self.search_space.vs[1]),
                      (self.search_space.theta[0], self.search_space.theta[1]),
                      (self.search_space.phi[0], self.search_space.phi[1])]
        elif self.wave_type == 'SH' and self.search_space.free_surface:
            bounds = [(self.search_space.vs[0], self.search_space.vs[1]),
                      (self.search_space.theta[0], self.search_space.theta[1]),
                      (self.search_space.phi[0], self.search_space.phi[1])]
        elif self.wave_type == 'R':
            bounds = [(self.search_space.vr[0], self.search_space.vr[1]),
                      (self.search_space.phi[0], self.search_space.phi[1]),
                      (self.search_space.xi[0], self.search_space.xi[1])]
        elif self.wave_type == 'L':
            bounds = [(self.search_space.vl[0], self.search_space.vl[1]),
                      (self.search_space.phi[0], self.search_space.phi[1])]
        else:
            sys.exit('No valid wave type in Estimator.solve()!')

        return bounds

    def _test_wave_grid(self, location):
        if (self.wave_type == 'P' or self.wave_type == 'SV') and self.search_space.free_surface:
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal,
                             vp=self.search_space.vp_vect[location[0]],
                             vs=self.search_space.vs_vect[location[1]], theta=self.search_space.theta_vect[location[2]],
                             phi=self.search_space.phi_vect[location[3]])
        elif self.wave_type == 'P' and not self.search_space.free_surface:
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal,
                             theta=self.search_space.theta_vect[location[0]],
                             phi=self.search_space.phi_vect[location[1]], free_surface=False)
        elif (self.wave_type == 'SV' or self.wave_type == 'SH') and not self.search_space.free_surface:
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal,
                             vs=self.search_space.vs_vect[location[0]],
                             theta=self.search_space.theta_vect[location[1]],
                             phi=self.search_space.phi_vect[location[2]], free_surface=False)
        elif self.wave_type == 'SH' and self.search_space.free_surface:
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal,
                             vs=self.search_space.vs_vect[location[0]],
                             theta=self.search_space.theta_vect[location[1]],
                             phi=self.search_space.phi_vect[location[2]])
        elif self.wave_type == 'R':
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal,
                             vr=self.search_space.vr_vect[location[0]],
                             phi=self.search_space.phi_vect[location[1]],
                             xi=self.search_space.xi_vect[location[2]])
        elif self.wave_type == 'L':
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal,
                             vl=self.search_space.vl_vect[location[0]],
                             phi=self.search_space.phi_vect[location[1]])
        else:
            sys.exit('Invalid wave type specified in Estimator object!')
        return test_wave

    def _test_wave_global(self, x):
        if (self.wave_type == 'P' or self.wave_type == 'SV') and self.search_space.free_surface:
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal, vp=x[0],
                             vs=x[1], theta=x[2],
                             phi=x[3])
        elif self.wave_type == 'P' and not self.search_space.free_surface:
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal, theta=x[0],
                             phi=x[1], free_surface=False)
        elif (self.wave_type == 'SV' or self.wave_type == 'SH') and not self.search_space.free_surface:
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal, vs=x[0],
                             theta=x[1],
                             phi=x[2], free_surface=False)
        elif self.wave_type == 'SH' and self.search_space.free_surface:
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal, vs=x[0],
                             theta=x[1],
                             phi=x[2])
        elif self.wave_type == 'R':
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal, vr=x[0],
                             phi=x[1],
                             xi=x[2])
        elif self.wave_type == 'L':
            test_wave = Wave(wave_type=self.wave_type, v_scal=self.search_space.v_scal, vl=x[0],
                             phi=x[1])
        else:
            sys.exit('Invalid wave type specified in Estimator object!')
        return test_wave


class Wave:
    """
    Attributes:
        wave_type: WAVE TYPE
            'P' : P-wave
            'SV': SV-wave
            'SH': SH-wave
            'L' : Love-wave
            'R' : Rayleigh-wave
        vp: P-wave velocity (m/s) at the receiver location
        vs: S-wave velocity (m/s) at the receiver location
        vl: Love-wave velocity (m/s)
        vr: Rayleigh-wave velocity (m/s)
        theta: Inclination (degree), only for body waves
        xi: Ellipticity angle (rad) for Rayleigh waves
        v_scal: scaling velocity (m/s) to make translations dimensionless, Default: 1 (m/s)
        free_surface: True (default): the wave is recorded at the free surface, False: the wave is recorded inside the medium
        polarization: 6-C polarization vector (automatically computed from the other attributes)

    Methods:
        polarization_vector: Computes the polarization vector for the current instance of the class 'Wave'
        theta_rad: Outputs the inclination angle in rad
        phi_rad: Outputs the azimuth angle in rad
    """

    def __init__(self, wave_type=None, vp=None, vs=None, vl=None, vr=None, theta=None, phi=None, xi=None, v_scal=1,
                 free_surface=True, isnone=False):
        self.free_surface, self.wave_type = free_surface, wave_type
        self.vp, self.vs, self.vl, self.vr, self.theta, self.phi, self.xi = vp, vs, vl, vr, theta, phi, xi
        self.v_scal = v_scal
        self.isnone = isnone
        self.polarization = self.polarization_vector

    @property
    def polarization_vector(self):
        """
        Computes the six-component polarization vector for a specified wave-type
        at the free surface according to Equations 40 in Sollberger et al. (2018)

        wave: Instance of the class 'Wave'
        """
        if self.isnone:
            polarization = np.asmatrix(
                [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
            return polarization

        if self.wave_type == 'P':
            if self.free_surface and 4 > self.vp / self.vs > 1.66:  # Exclude unphysical vp/vs ratios
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)

                v = (self.vp ** 2 - 2. * self.vs ** 2) / (2. * (self.vp ** 2 - self.vs ** 2))  # Poisson's ratio
                kappa = (2. * (1 - v) / (1 - 2 * v)) ** (1 / 2.)
                theta_s = np.arcsin((1 / kappa) * np.sin(theta_rad))  # angle of reflected S-wave

                # amplitude of reflected P-wave
                alpha_pp = (np.sin(2 * theta_rad) * np.sin(2 * theta_s) - kappa ** 2 * (np.cos(2 * theta_s)) ** 2) \
                           / (np.sin(2 * theta_rad) * np.sin(2 * theta_s) + kappa ** 2 * (np.cos(2 * theta_s)) ** 2)

                # amplitude of reflected S-wave
                alpha_ps = (2 * kappa * np.sin(2 * theta_rad) * np.cos(2 * theta_s)) \
                           / (np.sin(2 * theta_rad) * np.sin(2 * theta_s) + kappa ** 2 * (np.cos(2 * theta_s)) ** 2)

                v_x = -(np.sin(theta_rad) * np.cos(phi_rad)
                        + alpha_pp * np.sin(theta_rad) * np.cos(phi_rad)
                        + alpha_ps * np.cos(theta_s) * np.cos(phi_rad)) / self.v_scal
                v_y = -(np.sin(theta_rad) * np.sin(phi_rad)
                        + alpha_pp * np.sin(theta_rad) * np.sin(phi_rad)
                        + alpha_ps * np.cos(theta_s) * np.sin(phi_rad)) / self.v_scal
                v_z = -(np.cos(theta_rad)
                        - alpha_pp * np.cos(theta_rad)
                        + alpha_ps * np.sin(theta_s)) / self.v_scal
                w_x = (1 / 2.) * alpha_ps * np.sin(phi_rad) / self.vs
                w_y = -(1 / 2.) * alpha_ps * np.cos(phi_rad) / self.vs
                w_z = 0. * w_x

                polarization = np.asmatrix([v_x, v_y, v_z, w_x, w_y, w_z])
            elif not self.free_surface and self.vp / self.vs > 1.66:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)
                v_x = - (1. / self.v_scal) * np.sin(theta_rad) * np.cos(phi_rad)
                v_y = - (1. / self.v_scal) * np.sin(theta_rad) * np.sin(phi_rad)
                v_z = - (1. / self.v_scal) * np.cos(theta_rad)
                w_x = 0.
                w_y = 0.
                w_z = 0.
                polarization = np.asmatrix([v_x, v_y, v_z, w_x, w_y, w_z])
            else:
                polarization = np.asmatrix(
                    [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])

        elif self.wave_type == 'SV':
            if self.free_surface and 4 > self.vp / self.vs > 1.66:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)
                v = (self.vp ** 2 - 2. * self.vs ** 2) / (2. * (self.vp ** 2 - self.vs ** 2))  # poisson's ratio
                kappa = (2 * (1 - v) / (1 - 2 * v)) ** (1 / 2.)
                theta_crit = np.arcsin(1 / kappa)

                # Check whether the current incidence angle is at or above the critical angle
                if theta_rad == theta_crit:
                    theta_p = np.pi / 2.
                    alpha_sp = (4. * (kappa ** 2 - 1)) / (kappa * (2 - kappa ** 2))
                    alpha_ss = -1
                elif theta_crit < theta_rad < 0.9 * np.pi / 4:
                    # Incidence angles above the critical angle will yield a complex polarization
                    theta_p = np.arcsin(complex(np.sin(theta_rad) * self.vp / self.vs, 0))
                    alpha_ss = (4 * (np.sin(theta_rad) ** 2 - kappa ** (-2)) * np.sin(2 * theta_rad) ** 2 *
                                np.sin(theta_rad) ** 2
                                - np.cos(theta_rad) ** 4 + 4 * 1j * (np.sin(theta_rad) ** 2 - kappa ** -2) ** (1 / 2.) *
                                np.sin(2 * theta_rad) * np.sin(theta_rad) * (np.cos(2 * theta_rad)) ** 2) \
                               / (np.cos(2 * theta_rad) ** 4 + 4 * (np.sin(theta_rad) ** 2 - kappa ** -2) *
                                  np.sin(2 * theta_rad) ** 2 * np.sin(theta_rad) ** 2)
                    alpha_sp = (2 * kappa ** -1 * np.sin(2 * theta_rad) * np.cos(2 * theta_rad)
                                * (np.cos(2 * theta_rad) ** 2 - 2
                                   * 1j * (np.sin(theta_rad) ** 2 - kappa ** (-2)) ** (1 / 2.)
                                   * np.sin(2 * theta_rad) * np.sin(theta_rad))) / \
                               (np.cos(2 * theta_rad) ** 4 + 4 * (np.sin(theta_rad) ** 2 - kappa ** -2)
                                * np.sin(2 * theta_rad) ** 2 * np.sin(theta_rad) ** 2)

                elif theta_rad < theta_crit:

                    theta_p = np.arcsin(np.sin(theta_rad) * self.vp / self.vs)

                    alpha_ss = (np.sin(2 * theta_rad) * np.sin(2 * theta_p) - kappa ** 2 * (np.cos(2 * theta_p)) ** 2) \
                               / (np.sin(2 * theta_rad) * np.sin(2 * theta_p) + kappa ** 2 * (
                        np.cos(2 * theta_rad)) ** 2)
                    alpha_sp = -(kappa * np.sin(4 * theta_rad)) \
                               / (np.sin(2 * theta_rad) * np.sin(2 * theta_p)
                                  + kappa ** 2 * (np.cos(2 * theta_rad)) ** 2)
                else:
                    theta_p = float("nan")
                    alpha_ss = float("nan")
                    alpha_sp = float("nan")

                v_x = (np.cos(theta_rad) * np.cos(phi_rad)
                       - alpha_ss * np.cos(theta_rad) * np.cos(phi_rad)
                       - alpha_sp * np.sin(theta_p) * np.cos(phi_rad)) / self.v_scal
                v_y = (np.cos(theta_rad) * np.sin(phi_rad)
                       - alpha_ss * np.cos(theta_rad) * np.sin(phi_rad)
                       - alpha_sp * np.sin(theta_p) * np.sin(phi_rad)) / self.v_scal
                v_z = -(np.sin(theta_rad)
                        + alpha_ss * np.sin(theta_rad)
                        - alpha_sp * np.cos(theta_p)) / self.v_scal

                w_x = (1 / 2.) * (1 + alpha_ss) * np.sin(phi_rad) / self.vs
                w_y = -(1 / 2.) * (1 + alpha_ss) * np.cos(phi_rad) / self.vs
                w_z = 0. * w_x
                polarization = np.asmatrix([v_x, v_y, v_z, w_x, w_y, w_z])
                if theta_rad > 0.9 * np.pi / 4:
                    polarization = np.asmatrix(
                        [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
            elif not self.free_surface and 4 > self.vp / self.vs > 1.66:
                theta_rad = np.radians(self.theta)
                phi_rad = np.radians(self.phi)
                v_x = (1. / self.v_scal) * np.cos(theta_rad) * np.cos(phi_rad)
                v_y = (1. / self.v_scal) * np.cos(theta_rad) * np.sin(phi_rad)
                v_z = -(1. / self.v_scal) * np.sin(theta_rad)
                w_x = (2 * self.vs) ** -1 * np.sin(phi_rad)
                w_y = - (2 * self.vs) ** -1 * np.cos(phi_rad)
                w_z = 0.
                polarization = np.asmatrix([v_x, v_y, v_z, w_x, w_y, w_z])

            else:
                polarization = np.asmatrix(
                    [float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')])
        elif self.wave_type == 'SH':
            if self.free_surface:
                phi_rad = np.radians(self.phi)
                theta_rad = np.radians(self.theta)
                v_x = 2. / self.v_scal * np.sin(phi_rad)
                v_y = -2. / self.v_scal * np.cos(phi_rad)
                v_z = 0.
                w_x = 0.
                w_y = 0.
                w_z = 1. / self.vs * np.sin(theta_rad)
                polarization = np.asmatrix([v_x, v_y, v_z, w_x, w_y, w_z])
            else:
                phi_rad = np.radians(self.phi)
                theta_rad = np.radians(self.theta)
                v_x = (1. / self.v_scal) * np.sin(phi_rad)
                v_y = - (1. / self.v_scal) * np.cos(phi_rad)
                v_z = 0.
                w_x = - (2 * self.vs) ** -1 * np.cos(theta_rad) * np.cos(phi_rad)
                w_y = - (2 * self.vs) ** -1 * np.cos(theta_rad) * np.sin(phi_rad)
                w_z = - (2 * self.vs) ** -1 * np.sin(theta_rad)
                polarization = np.asmatrix([v_x, v_y, v_z, w_x, w_y, w_z])

        elif self.wave_type == 'R':
            phi_rad = np.radians(self.phi)
            v_x = 1j * 1. / self.v_scal * np.sin(self.xi) * np.cos(phi_rad)
            v_y = 1j * 1. / self.v_scal * np.sin(self.xi) * np.sin(phi_rad)
            v_z = -1. / self.v_scal * np.cos(self.xi)

            w_x = 1. / self.vr * np.sin(phi_rad) * np.cos(self.xi)
            w_y = -1. / self.vr * np.cos(phi_rad) * np.cos(self.xi)
            w_z = 0.
            polarization = np.asmatrix([v_x, v_y, v_z, w_x, w_y, w_z])
        elif self.wave_type == 'L':
            phi_rad = np.radians(self.phi)
            v_x = 1 / self.v_scal * np.sin(phi_rad)
            v_y = -1 / self.v_scal * np.cos(phi_rad)
            v_z = 0.

            w_x = 0.
            w_y = 0.
            w_z = 1. / (2 * self.vl)
            polarization = np.asmatrix([v_x, v_y, v_z, w_x, w_y, w_z])

        else:
            sys.exit("Invalid wave type specified in 'Wave' object!")

        polarization = np.divide(polarization, np.linalg.norm(polarization))
        polarization = polarization.T
        return polarization

    @property
    def theta_rad(self):
        return np.radians(self.theta)

    @property
    def phi_rad(self):
        return np.radians(self.phi)


class EstimatedWave(Wave):
    def __init__(self, wave_type=None, vp=None, vs=None, vl=None, vr=None, theta=None, phi=None, xi=None, v_scal=1,
                 free_surface=True, Lmax=None, L=None, isnone=False, dop=None):
        super().__init__(wave_type=wave_type, vp=vp, vs=vs, vl=vl, vr=vr, theta=theta, phi=phi, xi=xi, v_scal=v_scal,
                         free_surface=free_surface, isnone=isnone)
        self.free_surface, self.wave_type = free_surface, wave_type
        self.isnone = isnone
        self.vp, self.vs, self.vl, self.vr, self.theta, self.phi, self.xi = vp, vs, vl, vr, theta, phi, xi
        self.v_scal = v_scal
        # self.polarization = self.polarization_vector
        self.Lmax, self.L = Lmax, L
        self.dop = dop


class SearchSpace:

    def __init__(self, wave_type=None, vp=None, vs=None, vl=None, vr=None, theta=None, phi=None, xi=None,
                 free_surface=True, v_scal=1):
        """
        wave_type: WAVE TYPE
            'P' : P-wave
            'SV': SV-wave
            'SH': SH-wave
            'L' : Love-wave
            'R' : Rayleigh-wave
        vp = [vp_min, vp_max, increment]: P-wave velocity (m/s) at the receiver location
        vs = [vs_min, vs_max, increment]: S-wave velocity (m/s) at the receiver location
        vl = [vl_min, vl_max, increment]: Love-wave velocity (m/s)
        vr = [vr_min, vr_max, increment]: Rayleigh-wave velocity (m/s)
        theta = [theta_min, theta_max, increment] Inclination (degree), only for body waves
        xi = [xi_min, xi_max, increment]: Ellipticity angle (rad) for Rayleigh waves
        """

        self.wave_type = wave_type
        self.vp, self.vs, self.vl, self.vr, self.theta, self.phi, self.xi = vp, vs, vl, vr, theta, phi, xi
        self.free_surface = free_surface
        self.v_scal = v_scal
        assert isinstance(wave_type, str)

        self.vp_vect, self.vs_vect, self.theta_vect, self.phi_vect, self.vr_vect, self.vl_vect, self.xi_vect = [
            None, None, None, None, None, None, None]

        if (wave_type == 'P' or wave_type == 'SV') and self.free_surface:
            if self.vp is None or self.vs is None or self.theta is None or self.phi is None:
                sys.exit(
                    f'To set up a free-surface {wave_type}-wave search space, search parameters for Vp, Vs, Theta, '
                    'and Phi must be specified!')
            self.vp_vect = np.arange(self.vp[0], self.vp[1] + self.vp[2], self.vp[2])
            self.vs_vect = np.arange(self.vs[0], self.vs[1] + self.vs[2], self.vs[2])
            self.theta_vect = np.arange(self.theta[0], self.theta[1] + self.theta[2], self.theta[2])
            self.phi_vect = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])
            self.n_vp = self.vp_vect.size
            self.n_vs = self.vs_vect.size
            self.n_theta = self.theta_vect.size
            self.n_phi = self.phi_vect.size
            self.N = self.n_vp * self.n_vs * self.n_theta * self.n_phi

        elif wave_type == 'P' and not self.free_surface:
            if self.theta is None or self.phi is None:
                sys.exit('To set up a P-wave (inside the medium) search space, search parameters for Theta, '
                         'and Phi must '
                         'be specified!')
            self.theta_vect = np.arange(self.theta[0], self.theta[1] + self.theta[2], self.theta[2])
            self.phi_vect = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])
            self.n_theta = self.theta_vect.size
            self.n_phi = self.phi_vect.size
            self.N = self.n_phi * self.n_theta

        elif (wave_type == 'SV' or wave_type == 'SH') and not self.free_surface:
            if self.vs is None or self.theta is None or self.phi is None:
                sys.exit(f'To set up a {wave_type}-wave (inside the medium) search space, search parameters for Vs, '
                         f'Theta, and Phi must '
                         'be specified!')
            self.vs_vect = np.arange(self.vs[0], self.vs[1] + self.vs[2], self.vs[2])
            self.theta_vect = np.arange(self.theta[0], self.theta[1] + self.theta[2], self.theta[2])
            self.phi_vect = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])
            self.n_vs = self.vs_vect.size
            self.n_theta = self.theta_vect.size
            self.n_phi = self.phi_vect.size
            self.N = self.n_phi * self.n_vs * self.n_phi

        elif wave_type == 'SH' and self.free_surface:
            if self.vs is None or self.theta is None or self.phi is None:
                sys.exit('To set up an SH-wave search space, search parameters for Vs, Theta, and Phi must '
                         'be specified!')
            self.vs_vect = np.arange(self.vs[0], self.vs[1] + self.vs[2], self.vs[2])
            self.theta_vect = np.arange(self.theta[0], self.theta[1] + self.theta[2], self.theta[2])
            self.phi_vect = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])
            self.n_vs = self.vs_vect.size
            self.n_theta = self.theta_vect.size
            self.n_phi = self.phi_vect.size
            self.N = self.n_theta * self.n_vs * self.n_phi

        elif wave_type == 'R':
            if self.vr is None or self.phi is None or self.xi is None:
                sys.exit('To set up a Rayleigh-wave search space, search parameters for Vr, Phi, and Xi must '
                         'be specified!')
            self.vr_vect = np.arange(self.vr[0], self.vr[1] + self.vr[2], self.vr[2])
            self.phi_vect = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])
            self.xi_vect = np.arange(self.xi[0], self.xi[1] + self.xi[2], self.xi[2])
            self.n_xi = self.xi_vect.size
            self.n_vr = self.vr_vect.size
            self.n_phi = self.phi_vect.size
            self.N = self.n_xi * self.n_vr * self.n_phi

        elif wave_type == 'L':
            if self.vl is None or self.phi is None:
                sys.exit('To set up a Love-wave search space, search parameters for Vl, and Phi must '
                         'be specified!')
            self.vl_vect = np.arange(self.vl[0], self.vl[1] + self.vl[2], self.vl[2])
            self.phi_vect = np.arange(self.phi[0], self.phi[1] + self.phi[2], self.phi[2])
            self.n_vl = self.vl_vect.size
            self.n_phi = self.phi_vect.size
            self.N = self.n_vl * self.n_phi

        else:
            sys.exit('Invalid wave type specified in SearchSpace object!')


def _compute_spec(traN, traE, traZ, rotN, rotE, rotZ, kind='spec', fmin=0, fmax=50, window_n=None, nfft=None,
                  overlap=0.5, w0=10, nf=100, dsfacf=1):
    if kind == 'cwt':
        npts = traN.stats.npts
        dt = traN.stats.delta

        traN = cwt(traN.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)
        traE = cwt(traE.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)
        traZ = cwt(traZ.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)
        rotN = cwt(rotN.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)
        rotE = cwt(rotE.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)
        rotZ = cwt(rotZ.data, dt, w0=w0, nf=nf, fmin=fmin, fmax=fmax)

        t = np.linspace(0, dt * npts, npts)
        f = np.logspace(np.log10(fmin),
                        np.log10(fmax),
                        nf)
    elif kind == 'spec':
        # parameters chosen to resemble matplotlib.mlab.specgram defaults
        kwargs = {'nperseg': window_n,
                  'fs': traN.stats.sampling_rate,
                  'nfft': nfft,
                  'noverlap': int(window_n * overlap),
                  'mode': 'complex',
                  'scaling': 'density',
                  'window': 'hanning',
                  'detrend': False}

        f, t, traN = spectrogram(traN.data, **kwargs)
        f, t, traE = spectrogram(traE.data, **kwargs)
        f, t, traZ = spectrogram(traZ.data, **kwargs)
        f, t, rotN = spectrogram(rotN.data, **kwargs)
        f, t, rotE = spectrogram(rotE.data, **kwargs)
        f, t, rotZ = spectrogram(rotZ.data, **kwargs)

        # normalization for mode='complex' differs from 'psd'
        traN *= 2 ** 0.5
        traE *= 2 ** 0.5
        traZ *= 2 ** 0.5
        rotN *= 2 ** 0.5
        rotE *= 2 ** 0.5
        rotZ *= 2 ** 0.5

    elif kind == 'st':
        dt = traN.stats.delta

        traN, f = s_transform(traN.data, dsfacf=dsfacf)
        traE, f = s_transform(traE.data, dsfacf=dsfacf)
        traZ, f = s_transform(traZ.data, dsfacf=dsfacf)
        rotN, f = s_transform(rotN.data, dsfacf=dsfacf)
        rotE, f = s_transform(rotE.data, dsfacf=dsfacf)
        rotZ, f = s_transform(rotZ.data, dsfacf=dsfacf)
        npts = traN.shape[1]
        t = np.linspace(0, dt * npts, npts)
        f = f / dt
    else:
        raise ValueError(
            'unknown TF method: %s (allowed: spec, cwt)' % kind)

    return f, t, traN, traE, traZ, rotN, rotE, rotZ


def _calc_dop_windows(dop_specwidth, dop_winlen, dt, fmax, fmin, kind, nf,
                      nfft, overlap, winlen_sec, ttot, f):
    # Calculate width of smoothing windows for degree of polarization analysis
    if kind == 'spec':
        ntsum = int(dop_winlen / (winlen_sec * (1 - overlap)))
        df = 1. / (nfft * dt)
        nfsum = int(dop_specwidth / df)
        dsfact = max(1, ntsum // 2)
        dsfacf = max(1, nfsum // 2)
    elif kind == 'cwt':
        periods = 1. / np.logspace(np.log10(fmin), np.log10(fmax), nf)
        ntsum = np.array(dop_winlen * periods / dt, dtype=int)
        df = (fmax / fmin) ** (1. / nf)
        nfsum = int(np.log(np.sqrt(dop_specwidth)) / np.log(df))
        dsfacf = max(1, nfsum // 4)
        dsfact = max(1, int(dop_winlen / dt))
    elif kind == 'st':
        df = f[1] - f[0]
        periods = 1. / f[1:]
        periods = np.append(ttot, periods)
        ntsum = np.array(dop_winlen * periods / dt, dtype=int)
        nfsum = int(dop_specwidth / df)
        dsfact = 1
        dsfacf = 1
    else:
        raise ValueError(f"Invalid time-frequency decomposition method: {kind}")

    if type(ntsum) == int and ntsum < 1:
        ntsum = 1
    if nfsum < 0.5:
        nfsum = 1

    return nfsum, ntsum, dsfacf, dsfact


def s_transform(signal, dsfacf=1):
    """
    Computes the S-transform of the input signal
    David Sollberger, 2020

    Returns:
        signal_strans: S-transform of the signal with dimensions (next_pow_2(len(signal))/2+1, next_pow_2(len(signal)))
        f: Normalized frequency vector (divide by sampling interval dt to get frequency in Hz)
        dsfacf: Down-sampling factor in the frequency direction -> enables efficient computation for long signals

    Code is adapted from a Matlab implementation by Vincent Perron, ETHZ, Switzerland,  which is based on the
    Matlab implementation by  Kalyan S. Dash, IIT Bhubaneswar, India

    """
    n = signal.shape[0]
    n_half = int(np.floor(n / 2))
    odd = n % 2

    f = np.concatenate([np.arange(n_half + 1), np.arange(-n_half + 1 - odd, 0)]) / n
    signal_fft = np.fft.fft(signal, n=n)
    signal_fft.shape = (1, n)
    periods = 1. / f[dsfacf:int(n_half) + 1:dsfacf]
    periods.shape = (periods.shape[0], 1)
    w = 2 * np.pi * np.tile(f, (periods.shape[0], 1)) * np.tile(periods, (1, n))
    gaussian = np.exp((- w ** 2) / 2)
    hw = _toeplitz_red(signal_fft[:, 1:n_half + 1].T, signal_fft.T, dsfacf)
    signal_strans = np.fft.ifft(hw * gaussian, n=n)
    signal_strans_zero = np.mean(signal) * np.ones(n)
    signal_strans_zero.shape = (1, signal_strans_zero.shape[0])
    signal_strans = np.concatenate([signal_strans_zero, signal_strans])

    return signal_strans.conj(), np.insert(f[dsfacf:n_half + 1:dsfacf], 0, 0)


def _toeplitz_red(c, r, dsfacf):
    """
    Constructs a non-symmetric Toeplitz matrix with c as its first column and r its first row
    :param c: first column
    :param r: first row
    :param dsfacf: down-sampling factor
    :return: T (Toeplitz matrix)
    David Sollberger, 2020 (david.sollberger@gmail.com)
    """

    v = np.concatenate((np.flip(r[1:]), c.conj()), axis=0)
    index = np.arange(r.shape[0] - 1, -1, -1) - 1
    index.shape = (index.shape[0], 1)
    ind = np.arange(dsfacf, c.shape[0] + 1, dsfacf)
    ind.shape = (1, ind.shape[0])

    index = np.add(index, ind).squeeze()
    T = v[index].squeeze().T
    return T


def _R_matrix(dir):
    # h = np.concatenate(np.real(dir)=
    pass


def load(file):
    fid = open(file, 'rb')
    obj = pickle.load(fid)
    fid.close()
    return obj
