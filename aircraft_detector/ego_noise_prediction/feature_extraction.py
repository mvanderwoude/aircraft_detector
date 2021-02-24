"""
Short summary.

Extract the features of the ego-noise data...
"""
import os
import shutil

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt

from aircraft_detector.utils.utils import (
    retrieve_files,
    get_feature_directory_name,
    refresh_directory,
    print_verbose,
    load_spectrum_settings,
)
import aircraft_detector.utils.feature_helper as fh
import aircraft_detector.utils.plot_helper as ph


class FeatureExtraction:
    def __init__(self, root_directory, feature_settings=None):
        # set root directory
        self._dir_root = root_directory
        # set the missing feature settings to their defaults
        if feature_settings is None:
            feature_settings = {}
        self._feature = load_spectrum_settings(feature_settings)
        # derive root output directory (feature dataset) from parameters
        self._dir_root_set = os.path.join(
            self._dir_root,
            "Ego-Noise Prediction",
            "Parameter Sets",
            get_feature_directory_name(self._feature),
        )
        # verbosity
        self.verbose = True  # print when method is finished
        self.super_verbose = False  # print for every single file

    def split_mav_data(self, train_test_ratio=0.8, train_val_ratio=0.8):
        # get files
        dir_audio = os.path.join(self._dir_root, "Raw", "Mav", "Audio")
        files_audio = [
            os.path.join(dir_audio, f) for f in sorted(os.listdir(dir_audio))
        ]
        dir_states = os.path.join(self._dir_root, "Raw", "Mav", "States")
        files_states = [
            os.path.join(dir_states, f) for f in sorted(os.listdir(dir_states))
        ]

        # split files into train-val-test
        audio_train, audio_test, states_train, states_test = train_test_split(
            files_audio, files_states, train_size=train_test_ratio, random_state=42
        )
        audio_train, audio_val, states_train, states_val = train_test_split(
            audio_train, states_train, train_size=train_val_ratio, random_state=42
        )

        # Group the files
        files_train = [audio_train, states_train]
        files_val = [audio_val, states_val]
        files_test = [audio_test, states_test]
        files = [files_train, files_val, files_test]

        # Output directory for the split
        dir_root_out = os.path.join(self._dir_root, "Ego-Noise Prediction", "Dataset")
        # Loop over subsets and data types
        for i, subset in enumerate(["Train", "Val", "Test"]):
            for j, data in enumerate(["Audio", "States"]):
                # Output directory for subset, data
                dir_dest = os.path.join(dir_root_out, subset, data)
                refresh_directory(dir_dest)
                # Copy to destination
                for f in files[i][j]:
                    shutil.copy(f, dir_dest)

    def extract_spectra(self, offset=50, scaling=80):
        # Loop over subsets
        for subset in ["Train", "Val", "Test"]:
            # Get audio files
            dir_audio = os.path.join(
                self._dir_root, "Ego-Noise Prediction", "Dataset", subset, "Audio"
            )
            files_audio = retrieve_files(dir_audio)
            # directory for the unsynchronized spectra
            dir_output = os.path.join(
                self._dir_root_set, "Unsynchronized", subset, "Spectra"
            )
            # Refresh directory
            refresh_directory(dir_output)

            # Loop through files in set
            for f in files_audio:
                # Extract spectrum
                Z = fh.extract_spectrum(f, self._feature)
                # Scale spectrum
                Z += offset
                Z /= scaling
                # Save to appropriate directory
                fn = os.path.split(f)[-1].replace(".wav", ".csv")
                fp = os.path.join(dir_output, fn)
                pd.DataFrame(Z).to_csv(fp, index=False, header=False)

                print_verbose(
                    self.super_verbose,
                    "Finished extracting feature for '%s' set." % subset,
                )

    def extract_states(self):
        # Loop over subsets
        for subset in ["Train", "Val", "Test"]:
            # Get states files
            dir_states = os.path.join(
                self._dir_root, "Ego-Noise Prediction", "Dataset", subset, "States"
            )
            files_states = retrieve_files(dir_states)
            # Directory for the unsynchronized states
            dir_output = os.path.join(
                self._dir_root_set, "Unsynchronized", subset, "States"
            )
            refresh_directory(dir_output)

            # Loop through files in set
            for f in files_states:  # xyz in NED frame
                # Read in as dataframe
                df = pd.read_csv(f, header=0)
                # Add delta-rpm
                df["rpm_1_delta"] = np.diff(df["rpm_1"].to_numpy(), prepend=0)
                df["rpm_2_delta"] = np.diff(df["rpm_2"].to_numpy(), prepend=0)
                df["rpm_3_delta"] = np.diff(df["rpm_3"].to_numpy(), prepend=0)
                df["rpm_4_delta"] = np.diff(df["rpm_4"].to_numpy(), prepend=0)
                # Add delta-cmd
                df["cmd_thrust_delta"] = np.diff(df["cmd_thrust"].to_numpy(), prepend=0)
                df["cmd_roll_delta"] = np.diff(df["cmd_roll"].to_numpy(), prepend=0)
                df["cmd_pitch_delta"] = np.diff(df["cmd_pitch"].to_numpy(), prepend=0)
                df["cmd_yaw_delta"] = np.diff(df["cmd_yaw"].to_numpy(), prepend=0)
                # Prune horizontal position
                df.drop(columns=["pos_x", "pos_y"], inplace=True)
                # Negate vertical position to get height
                df.rename(columns={"pos_z": "height"}, inplace=True)
                df["height"] *= -1
                # Replace north- and east velocities with magnitude (horizontal)
                df["vel_hor"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2)
                df.drop(columns=["vel_x", "vel_y"], inplace=True)
                # Negate downwards velocity go get vertical velocity
                df.rename(columns={"vel_z": "vel_ver"}, inplace=True)
                df["vel_ver"] *= -1
                # Replace north- and east accelerations with magnitude (horizontal)
                df["acc_hor"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2)
                df.drop(columns=["acc_x", "acc_y"], inplace=True)
                # Negate downwards velocity go get vertical acceleration
                df.rename(columns={"acc_z": "acc_ver"}, inplace=True)
                df["acc_ver"] *= -1
                # Re-order the frame
                cols = [
                    "delta_t",
                    "rpm_1",
                    "rpm_2",
                    "rpm_3",
                    "rpm_4",
                    "rpm_1_delta",
                    "rpm_2_delta",
                    "rpm_3_delta",
                    "rpm_4_delta",
                    "cmd_thrust",
                    "cmd_roll",
                    "cmd_pitch",
                    "cmd_yaw",
                    "cmd_thrust_delta",
                    "cmd_roll_delta",
                    "cmd_pitch_delta",
                    "cmd_yaw_delta",
                    "height",
                    "vel_hor",
                    "vel_ver",
                    "acc_hor",
                    "acc_ver",
                    "angle_phi",
                    "angle_theta",
                    "angle_psi",
                    "rate_p",
                    "rate_q",
                    "rate_r",
                ]
                df = df[cols]
                # Export
                fn = os.path.split(f)[-1]
                df.to_csv(os.path.join(dir_output, fn), header=True, index=False)

    def synchronize_data(self, skip_takeoff=True):

        # Loop over subsets
        for subset in ["Train", "Val", "Test"]:
            # list unsynchronized spectra
            dir_spectra = os.path.join(
                self._dir_root_set, "Unsynchronized", subset, "Spectra"
            )
            files_spectra = retrieve_files(dir_spectra)
            # list unsynchronized states
            dir_states = os.path.join(
                self._dir_root_set, "Unsynchronized", subset, "States"
            )
            files_states = retrieve_files(dir_states)

            # set the root output directory and refresh the output directories
            dir_root_output = os.path.join(self._dir_root_set, "Dataset", subset)
            refresh_directory(os.path.join(dir_root_output, "Spectra"))
            refresh_directory(os.path.join(dir_root_output, "States"))

            # synchronize each pair of files
            for i in range(len(files_spectra)):
                self._synchronize_pair(
                    files_spectra[i], files_states[i], dir_root_output, skip_takeoff
                )

    def _synchronize_pair(
        self, file_spectrum, file_states, dir_root_output, skip_takeoff
    ):
        # load spectrum
        Z = pd.read_csv(file_spectrum, header=None).to_numpy()
        # get time vector from the spectrum
        t_mic = librosa.times_like(
            Z,
            sr=self._feature["fft_sample_rate"],
            hop_length=self._feature["stft_hop_length"],
        )
        # load states
        S = pd.read_csv(file_states, header=0)
        # get time vector from state data
        t_mav = S["delta_t"].to_numpy()
        # scale the states and transpose: (time, states) -> (states, time)
        S.drop(columns=["delta_t"], inplace=True)
        S = fh.scale_states(S).to_numpy().transpose()

        # pair each time in t_mic with the closest found in t_mav
        assert len(t_mic) < len(t_mav)  # requires hop_size / fft_freq > 100
        idx = np.searchsorted(t_mav, t_mic)
        # only keep the matched pairs from t_mav
        S = S[:, idx]

        if skip_takeoff:
            # identify take off by delta-rpm spike (ROW 4-7)
            delta_rpms = S[4:8]
            max_spike = abs(delta_rpms).max()
            _, spikes = np.where(
                abs(delta_rpms) > 0.5 * max_spike
            )  # get column idx only
            last_spike = spikes.max()  # last column
            # cut off spikes + buffer frames
            buf = 5
            S = S[:, last_spike + 1 + buf :]
            Z = Z[:, last_spike + 1 + buf :]

        # filename of output identical to input
        fn = os.path.split(file_states)[-1]
        # export the synchronized spectra
        fp = os.path.join(dir_root_output, "Spectra", fn)
        pd.DataFrame(Z).to_csv(fp, header=False, index=False)
        # export the synchronized states
        fp = os.path.join(dir_root_output, "States", fn)
        pd.DataFrame(S).to_csv(fp, header=False, index=False)

        print_verbose(
            self.super_verbose,
            "Synchronized '%s' (%d datapoints)" % (fn, min(len(t_mic), len(t_mav))),
        )

    def plot_features(self, subset, states=None, idx=0):

        # plot all states by default
        if states is None:
            states = [
                "rpm",
                "rpm_delta",
                "cmd",
                "cmd_delta",
                "height",
                "vel",
                "acc",
                "angles",
                "rates",
            ]

        # set up figure
        n_rows = len(states) + 1
        fig = plt.figure(figsize=(8, 3 * n_rows), constrained_layout=True)
        gs = fig.add_gridspec(n_rows, 1)

        # load spectrum
        dir_spectrum = os.path.join(self._dir_root_set, "Dataset", subset, "Spectra")
        files_spectrum = retrieve_files(dir_spectrum)
        Z = pd.read_csv(files_spectrum[idx], header=None).to_numpy()
        # set plot title
        if self._feature["feature"] == "Mfcc":
            title = "MFCC (%d bins)" % self._feature["mfc_coefficients"]
        elif self._feature["feature"] == "Stft":
            title = "Spectrogram (%d frequency bins)" % self._feature["frequency_bins"]
        else:
            title = "%s-spectrogram (%d frequency bins)" % (
                self._feature["feature"],
                self._feature["frequency_bins"],
            )
        # plot spectrum
        ax = fig.add_subplot(gs[0])
        ax.set_title(title)
        ax = ph.plot_spectrum(Z, self._feature)

        # load states
        dir_states = os.path.join(self._dir_root_set, "Dataset", subset, "States")
        files_states = retrieve_files(dir_states)
        S = pd.read_csv(files_states[idx], header=None).to_numpy()
        # plot relevant states
        colors = ["orangered", "darkolivegreen", "steelblue", "goldenrod"]
        for i, state_name in enumerate(states):
            ax = fig.add_subplot(gs[1 + i])
            ax = ph.plot_states_synchronized(S, state_name, self._feature, colors)

        plt.show()
        return fig
