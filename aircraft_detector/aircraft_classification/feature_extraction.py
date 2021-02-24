import os
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

import aircraft_detector.utils.feature_helper as fh
import aircraft_detector.utils.plot_helper as ph

from aircraft_detector.utils.utils import (
    retrieve_files,
    get_feature_directory_name,
    refresh_directory,
    print_verbose,
    load_spectrum_settings,
)


class FeatureExtraction:
    """Extracts the features (spectra) required for aircraft classification.

    Expects the following parameters (if none supplied, defaults are used):
    features_to_extract -- iterable containing features to be extracted:
        'Stft': (regular spectrogram),
        'Mel': (Mel spectrogram),
        'Cqt': (Constant-Q transform),
        'Mfcc': (Mel frequency ceptral coefficients).
        By default, all features above are extracted.
    parameter_dict -- dictionary containing parameters for spectra generation:
        'fft_sample_rate': target sample rate (default: 44.1kHz),
        'stft_hop_length': hop length of the STFT (default: 512),
        'stft_window_length': window length of the STFT (default: 1024),
        'frequency_bins': number of frequency bins of the STFT (default: 60),
        'cqt_min_frequency': lowest tonal frequency of the cqt (default: 'C1'),
        'mfc_coefficients': number of coefficients for the MFCC (default: 13).
    data_augmentations -- iterable containing desired data augmentations:
        'Pitch Shift': shifts the pitch of the audio by [-2, -1, 1, 2] octaves,
        'Time Stretch': stretches audio by a factor [0.7, 0.85, 1.15, 1.30],
        'Class Mix': intra-mixing of samples within the same class.
        By default, all augmentations are extracted.
    Other considerations:
        if 'Cqt' features are to be extracted, 'frequency_bins' must be a
        multiple of 12.
        'mfc_coefficients' may not be larger than 'frequency_bins'
        'fft_sample_rate' divided by 'stft_hop_length' must not be larger than
        100 Hz (default: 86.13 Hz).
        All parameters are case sensitive.
    Example usage of class:
        features = ['Stft', 'Mel']
        parameters = {"fft_sample_rate": 36000, "frequency_bins": 64}
        augmentations = ['Pitch Shift']
        extractor = FeatureExtraction(features, parameters, augmentations)
        extractor.create_clean_dataset()
        extractor.plot_spectra_clean('Test')
        extractor.create_noisy_dataset([0.25, 0.50, 1.00])
        extractor.create_denoised_test_set([0.25, 0.50, 1.00])
    """

    def __init__(
        self, root_directory, feature_settings=None,
    ):
        # set root directory
        self._dir_root = root_directory
        # set the missing feature settings to their defaults
        if feature_settings is None:
            feature_settings = {}
        self._feature = load_spectrum_settings(feature_settings)

        # root directory (input) containing the ESC50 audio
        self._dir_root_audio = os.path.join(
            self._dir_root, "Aircraft Classification", "Audio"
        )
        # root directory (input) where the ego-noise prediction features are stored
        self._dir_root_enp = os.path.join(
            self._dir_root,
            "Ego-Noise Prediction",
            "Parameter Sets",
            get_feature_directory_name(self._feature),
        )
        # root directory (output) to store the aircraft classification features
        self._dir_root_ac = os.path.join(
            self._dir_root,
            "Aircraft Classification",
            "Parameter Sets",
            get_feature_directory_name(self._feature),
        )

        # verbosity
        self.verbose = True
        self.super_verbose = False

    def create_clean_dataset(self, augmentations=None, overwrite=False):
        """Create (extract and export) the clean dataset from the audio data.

        Keyword argument:
            overwrite -- whether to overwrite existing data (default: False).
        """
        # get list of dataset directory names
        if augmentations is not None:
            subsets = ["Train", "Val", "Test"]
            subsets += ["Train " + aug for aug in augmentations]
        else:
            # extract all sets available
            subsets = sorted(os.listdir(self._dir_root_audio))

        for subset in subsets:
            # output directory (spectra)
            dir_output = os.path.join(
                self._dir_root_ac, "Features", "Clean", "Spectra", subset
            )
            # only extract feature if set does not exist or should be overwritten
            if os.path.exists(dir_output) and not overwrite:
                continue
            refresh_directory(dir_output)

            # get audio files (ESC-50)
            dir_input = os.path.join(self._dir_root_audio, subset)
            files_input = retrieve_files(dir_input)

            # multiprocessing
            part = partial(self._export_spectrum, dir_out=dir_output)
            with Pool(processes=os.cpu_count() - 1) as pool:
                pool.map(part, files_input)

    def _export_spectrum(self, file, dir_out):
        """Extract and export the generated spectrum,."""
        # extract spectrum, then scale, then extract delta
        Z = fh.extract_spectrum(file, self._feature)
        Z, _ = fh.normalize_spectrum(Z)
        dZ = librosa.feature.delta(Z, mode="mirror")
        Z = np.concatenate((Z, dZ), axis=0)

        # export feature (keep filename)
        filename = os.path.split(file)[-1].replace(".wav", ".csv")
        filepath_out = os.path.join(dir_out, filename)
        pd.DataFrame(Z).to_csv(filepath_out, index=False, header=False)

    def create_mixed_dataset(
        self, noise_ratio, augmentations=None, seed=42, overwrite=False
    ):
        """Mix clean spectra with ego-noise spectra at the specified ratios.

        Keyword arguments:
            noise_ratios -- iterable containing the ratio of the noise
            compared to the signal (default: [0.5, 1.0]),
            overwrite -- whether to overwrite existing data (default: False).
        Note that the mixing is entirely random. When generating new features
        (i.e. deciding to extract both 'Mel' and 'Stft' features instead of
        only 'Stft'), it is recommended to set overwrite to True to ensure
        mixing consistency between existing and new features.
        """
        # get list of dataset directory names
        if augmentations is not None:
            subsets = ["Train", "Val", "Test"]
            subsets += ["Train " + aug for aug in augmentations]
        else:
            # extract all sets available
            subsets = os.listdir(self._dir_root_audio)

        # loop through sets in dataset
        for subset in subsets:
            # retrieve 'clean' spectra (ESC-50)
            dir_clean = os.path.join(
                self._dir_root_ac, "Features", "Clean", "Spectra", subset
            )
            files_clean = retrieve_files(dir_clean)
            # retrieve ego-noise spectra (MAV)
            subset_enp = subset.split(" ")[0]  # use 'Train' set for augmentations
            dir_noise = os.path.join(
                self._dir_root_enp, "Dataset", subset_enp, "Spectra"
            )
            files_noise = retrieve_files(dir_noise)
            # retrieve states belonging to ego-noise
            dir_states = os.path.join(
                self._dir_root_enp, "Dataset", subset_enp, "States"
            )
            files_states = retrieve_files(dir_states)

            # generate a list of directory-specific 'seeds' from the given seed
            # to preserve reproducible randomness while multiprocessing
            dir_seed = seed + len(subset)
            np.random.seed(dir_seed)
            seeds = np.random.randint(0, 10 * len(files_clean), len(files_clean))

            # make state dir. if it does not exist (do not overwrite, ever)
            dir_out = os.path.join(
                self._dir_root_ac, "Features", "Mixed", "States", subset
            )
            if not os.path.exists(dir_out):
                os.makedirs(dir_out)

            # set output directory (spectra)
            dir_out = os.path.join(
                self._dir_root_ac,
                "Features",
                "Mixed",
                "Spectra",
                "Ratio_%.2f" % noise_ratio,
                subset,
            )
            # check if it exists or should be overwritten
            if os.path.exists(dir_out) and not overwrite:
                continue  # skip set
            refresh_directory(dir_out)

            # set up multiprocessing to mix audio
            part = partial(
                self._create_mixed_spectrum,
                files_noise=files_noise,
                files_states=files_states,
                ratio=noise_ratio,
                subset=subset,
            )
            with Pool(processes=os.cpu_count() - 1) as pool:
                pool.starmap(part, list(zip(files_clean, seeds)))

    def create_mixed_test_set(self, noise_ratio, seed=42, overwrite=False):
        # make state dir. if it does not exist (do not overwrite, ever)
        dir_out = os.path.join(self._dir_root_ac, "Features", "Mixed", "States", "Test")
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        # set output directory (spectra)
        dir_out = os.path.join(
            self._dir_root_ac,
            "Features",
            "Mixed",
            "Spectra",
            "Ratio_%.2f" % noise_ratio,
            "Test",
        )
        # check if it exists or should be overwritten
        if os.path.exists(dir_out) and not overwrite:
            return
        refresh_directory(dir_out)

        # retrieve 'clean' spectra (ESC-50)
        dir_clean = os.path.join(
            self._dir_root_ac, "Features", "Clean", "Spectra", "Test"
        )
        files_clean = retrieve_files(dir_clean)
        # retrieve ego-noise spectra (MAV)
        dir_noise = os.path.join(self._dir_root_enp, "Dataset", "Test", "Spectra")
        files_noise = retrieve_files(dir_noise)
        # retrieve states belonging to ego-noise
        dir_states = os.path.join(self._dir_root_enp, "Dataset", "Test", "States")
        files_states = retrieve_files(dir_states)

        # generate a list of directory-specific 'seeds' from the given seed
        # to preserve reproducible randomness while multiprocessing
        dir_seed = seed + len("Test")
        np.random.seed(dir_seed)
        seeds = np.random.randint(0, 10 * len(files_clean), len(files_clean))

        # set up multiprocessing to mix audio
        part = partial(
            self._create_mixed_spectrum,
            files_noise=files_noise,
            files_states=files_states,
            ratio=noise_ratio,
            subset="Test",
        )
        with Pool(processes=os.cpu_count() - 1) as pool:
            pool.starmap(part, list(zip(files_clean, seeds)))

    def _create_mixed_spectrum(
        self,
        file_clean,
        seed,
        files_noise,
        files_states,
        ratio,
        subset,
        max_context_frames=10,
    ):
        # reset seed
        np.random.seed(seed)

        # open clean file (source)
        C = pd.read_csv(file_clean, header=None).to_numpy()
        C = C[: C.shape[0] // 2]  # omit deltas, recompute them after mixing

        # select a random MAV file index
        idx_file = np.random.randint(low=0, high=len(files_noise))
        # open noise file and corresponding states file
        file_noise = files_noise[idx_file]
        N = pd.read_csv(file_noise, header=None).to_numpy()
        file_states = files_states[idx_file]
        S = pd.read_csv(file_states, header=None).to_numpy()

        # get the number of frames in the clean file
        n_frames = C.shape[1]
        # select random index from noise file to sample from
        idx_frame = np.random.randint(
            low=max_context_frames, high=S.shape[1] - n_frames
        )
        # slice noise and state files to match length of clean file
        N = N[:, idx_frame : idx_frame + n_frames]
        S = S[:, idx_frame : idx_frame + n_frames]

        # mix at noise ratio and add deltas
        M = C + ratio * N  # * (1 / (1 + ratio))
        # M, _ = fh.normalize_spectrum(M)
        dM = librosa.feature.delta(M, mode="mirror")
        M = np.concatenate((M, dM), axis=0)

        # export the mix
        fn = os.path.split(file_clean)[-1]  # same filename as clean
        fp = os.path.join(
            self._dir_root_ac,
            "Features",
            "Mixed",
            "Spectra",
            "Ratio_%.2f" % ratio,
            subset,
            fn,
        )
        pd.DataFrame(M).to_csv(fp, header=False, index=False)

        # export the states corresponding to the noise used to mix
        fp = os.path.join(
            self._dir_root_ac,
            "Features",
            "Mixed",
            "States",
            subset,
            "%s_%d_%d.csv" % (fn.split(".")[0], idx_file, idx_frame),
        )  # add file and frame idx from noise file to filename
        pd.DataFrame(S).to_csv(fp, header=False, index=False)

    def create_denoised_test_set(self, noise_ratio, enp_model_index=0, overwrite=False):
        """Denoise the mixed spectra to obtain denoised test sets.

        Keyword arguments:
            noise_ratios -- iterable containing the ratio of the noise
            compared to the signal (default: [0.5, 1.0]),
            enp_model_index -- selects which ego-noise predictor (enp) to use
            for denoising, if multiple are available (default: 0),
            overwrite -- whether to overwrite existing data (default: False).

        Note: only noise ratios that were used for the mixing of noisy data
        (via create_noisy_dataset) can be used for denoising.
        """
        # set output directory
        dir_out = os.path.join(
            self._dir_root_ac,
            "Features",
            "Denoised",
            "Spectra",
            "Ratio_%.2f" % noise_ratio,
            "Test",
        )
        # check if it exists or should be overwritten
        if os.path.exists(dir_out) and not overwrite:
            return
        refresh_directory(dir_out)

        # get directory containing noise model
        dir_model_enp = sorted(os.listdir(os.path.join(self._dir_root_enp, "Models")))[
            enp_model_index
        ]
        # get context frames !!! get from config instead...
        model_context = int(dir_model_enp.split("_")[-1][-1])
        # load files containing predicted noise
        dir_predicted = os.path.join(
            self._dir_root_enp, "Models", dir_model_enp, "Output", "Test", "Predicted",
        )
        files_predicted = retrieve_files(dir_predicted)

        # load states files
        dir_states = os.path.join(
            self._dir_root_ac, "Features", "Mixed", "States", "Test"
        )
        files_states = retrieve_files(dir_states)

        # load mixed files
        dir_mix = os.path.join(
            self._dir_root_ac,
            "Features",
            "Mixed",
            "Spectra",
            "Ratio_%.2f" % noise_ratio,
            "Test",
        )
        files_mix = retrieve_files(dir_mix)

        # set up pool
        part = partial(
            self._create_denoised_feature,
            files_predicted=files_predicted,
            files_states=files_states,
            context=model_context,
            ratio=noise_ratio,
            dir_out=dir_out,
        )
        with Pool(processes=os.cpu_count() - 1) as pool:
            pool.map(part, files_mix)

    def create_denoised_train_augmentation_set(
        self, noise_ratio, enp_model_index=0, overwrite=False
    ):
        """Denoise the mixed spectra to obtain a single denoised training set.

        Keyword arguments:
            noise_ratio -- the ratio of the noise compared to the signal
            (default: 1.0),
            enp_model_index -- selects which ego-noise predictor (enp) to use
            for denoising, if multiple are available (default: 0),
            overwrite -- whether to overwrite existing data (default: False).
        """
        # set output directory
        dir_out = os.path.join(
            self._dir_root_ac, "Features", "Clean", "Spectra", "Train Denoised",
        )
        # check if it exists or should be overwritten
        if os.path.exists(dir_out) and not overwrite:
            return  # exit
        refresh_directory(dir_out)

        # get directory containing noise model
        dir_model_enp = sorted(os.listdir(os.path.join(self._dir_root_enp, "Models")))[
            enp_model_index
        ]
        # get context frames !!! get from config instead...
        model_context = int(dir_model_enp.split("_")[-1][-1])
        # load files containing predicted noise
        dir_predicted = os.path.join(
            self._dir_root_enp, "Models", dir_model_enp, "Output", "Train", "Predicted",
        )
        files_predicted = retrieve_files(dir_predicted)

        # load states files
        dir_states = os.path.join(
            self._dir_root_ac, "Features", "Mixed", "States", "Train"
        )
        files_states = retrieve_files(dir_states)

        # load mixed files
        dir_mix = os.path.join(
            self._dir_root_ac,
            "Features",
            "Mixed",
            "Spectra",
            "Ratio_%.2f" % noise_ratio,
            "Train",
        )
        files_mix = retrieve_files(dir_mix)

        # set up pool
        part = partial(
            self._create_denoised_feature,
            files_predicted=files_predicted,
            files_states=files_states,
            context=model_context,
            ratio=noise_ratio,
            dir_out=dir_out,
        )
        with Pool(processes=os.cpu_count() - 1) as pool:
            pool.map(part, files_mix)

        # add extension to filenames to match other augmentations
        for fn in sorted(os.listdir(dir_out)):
            fn_new = "%s-dn.csv" % fn.split(".")[0]
            os.rename(os.path.join(dir_out, fn), os.path.join(dir_out, fn_new))

    def _create_denoised_feature(
        self, file_mix, files_predicted, files_states, context, ratio, dir_out
    ):
        """Denoises a mixed signal using the predicted noise."""
        # read in mixed file, omit deltas
        M = pd.read_csv(file_mix, header=None).to_numpy()
        M = M[: M.shape[0] // 2]
        n_frames = M.shape[1]

        # get filename of states file belonging to mixed file
        fn_mix = os.path.split(file_mix)[-1]
        fn_mix_stub = fn_mix.replace(".csv", "")
        fn_states = os.path.split([f for f in files_states if fn_mix_stub in f][0])[-1]
        fn_states_stub = fn_states.replace(".csv", "")
        # get file idx, frame idx from states filename
        idx_file = int(fn_states_stub.split("_")[-2])
        idx_frame = int(fn_states_stub.split("_")[-1])

        # read in predicted file at selected index,
        # also subtract context offset
        file_pred = files_predicted[idx_file]
        P = pd.read_csv(file_pred, header=None).to_numpy()[
            :, idx_frame - context : idx_frame - context + n_frames
        ]

        # get denoised spectrum
        # N is essentially multiplied by (1/(1+ratio)) when scaling the mix
        # D = M * (1 + ratio) - ratio * P
        D = M - ratio * P
        # rescale mix to [0, 1]
        # D, _ = fh.normalize_spectrum(D)
        # re-extract time deltas
        dD = librosa.feature.delta(D, mode="mirror")
        D = np.concatenate((D, dD), axis=0)

        # export spectrum
        fp_spectrum = os.path.join(dir_out, "%s.csv" % fn_mix_stub)
        pd.DataFrame(D).to_csv(fp_spectrum, header=False, index=False)

    def plot_spectra_clean(self, set_name, categories=None, idx=0):
        """Plot a selection of audio and spectra of a noise-free dataset.
            !!! fix featuers
        Keyword arguments:
            set_name -- set to be plotted (e.g. 'Test'),
            features -- iterable containing the features to be plotted
            (default: all),
            categories -- iterable containing the original categories
            (airplane, engine, etc.) to be plotted (default: all),
            idx -- index of the feature selected for plotting (default: 0).
        Example usage:
            plot_spectra_clean('Train Pitch Shift', ['airplane', 'wind'],
                               ['Mel'])
            This plots the first Melspectrum of the airplane and wind
            categories in the Pitch Shift augmentation set.
        This function does not verify whether the desired data exists, e.g. it
        does not know whether 'Pitch Shift' augmentation is used or if
        Melspectra were extracted.
        """
        # list all audio files
        dir_audio = os.path.join(
            self._dir_root, "Aircraft Classification", "Audio", set_name
        )
        files_audio = [
            os.path.join(dir_audio, f) for f in sorted(os.listdir(dir_audio))
        ]
        # plot all categories if not given
        if categories is None:
            file_categories = [os.path.split(f)[-1].split("_")[0] for f in files_audio]
            categories = sorted(list(set(file_categories)))

        # setup figure, subfigures
        n_categories = len(categories)
        fig = plt.figure(figsize=(6 * n_categories, 4), constrained_layout=False)
        gs = fig.add_gridspec(1, n_categories)

        # loop through categories
        for i, cat in enumerate(categories):
            # load audio file
            file_audio = [
                f for f in files_audio if os.path.split(f)[-1].split("_")[0] == cat
            ][idx]
            fn_stub = os.path.split(file_audio)[-1].split(".")[0]
            # prepare grid (plot audio half the size of features)
            gs_in = gs[i].subgridspec(3, 1, hspace=0.0, height_ratios=[1] + [2] * 2,)

            # plot audio
            ax = fig.add_subplot(gs_in[0])
            ph.plot_audio(file_audio, sr=self._feature["fft_sample_rate"])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(fn_stub)

            # load feature
            file_feature = os.path.join(
                self._dir_root_ac,
                "Features",
                "Clean",
                "Spectra",
                set_name,
                fn_stub + ".csv",
            )
            Z = pd.read_csv(file_feature, header=None).to_numpy()

            # plot spectrum
            ax = fig.add_subplot(gs_in[1])
            ph.plot_spectrum(Z[: Z.shape[0] // 2], self._feature)
            ax.get_xaxis().set_visible(False)
            # only keep left y-axis visible
            if i != 0:
                ax.get_yaxis().set_visible(False)

            # plot delta-spectrum
            ax = fig.add_subplot(gs_in[2])
            ph.plot_spectrum(Z[Z.shape[0] // 2 :], self._feature, colormap="coolwarm")
            ax.get_yaxis().set_visible(False)

    def plot_spectra_mixed(
        self,
        set_name,
        noise_ratio,
        categories=None,
        idx=0,
        plot_clean=True,
        plot_noise=True,
    ):
        """Plot a selection of spectra of a mixed dataset.

        Keyword arguments:
            set_name -- set to be plotted (e.g. 'Test'),
            noise_ratio -- the ratio of the noise compared to the signal,
            categories -- iterable containing the original categories
            (airplane, engine, etc.) to be plotted (default: all),
            features -- iterable containing the features to be plotted
            (default: all),
            idx -- index of the feature selected for plotting (default: 0).
            plot_clean -- whether to plot the clean spectra (default: True),
            plot_noise -- whether to plot the noise-only spectra
            (default: True)
            colorbar -- whether to plot a colorbar next to each plotted
            spectrum (default: False)
        Example usage:
            plot_spectra_mixed('Train', )...

        """
        # define directories
        dir_mixed = os.path.join(
            self._dir_root_ac,
            "Features",
            "Mixed",
            "Spectra",
            "Ratio_%.2f" % noise_ratio,
            set_name,
        )
        if plot_clean:
            dir_clean = os.path.join(
                self._dir_root_ac, "Features", "Clean", "Spectra", set_name
            )
        if plot_noise:
            # load noise files
            dir_noise = os.path.join(
                self._dir_root_enp, "Dataset", set_name.split(" ")[0], "Spectra"
            )
            files_noise = retrieve_files(dir_noise)
            # also load states files to recover noise indices
            dir_states = os.path.join(
                self._dir_root_ac, "Features", "Mixed", "States", set_name
            )
            files_states = retrieve_files(dir_states)

        # plot all categories if not given
        filenames = [f for f in sorted(os.listdir(dir_mixed))]
        if categories is None:
            file_categories = [f.split("_")[0] for f in filenames]
            categories = sorted(list(set(file_categories)))

        # setup figure, subfigures
        n_categories = len(categories)
        n_variants = 1 + int(plot_clean) + int(plot_noise)
        fig = plt.figure(
            figsize=(6 * n_categories, 4 * n_variants), constrained_layout=False,
        )
        gs = fig.add_gridspec(n_variants * 2, n_categories)

        # loop through categories
        for i, cat in enumerate(categories):
            # get filename
            fn = [f for f in filenames if f.split("_")[0] == cat][idx]
            # load mixed file
            file_mixed = os.path.join(dir_mixed, fn)
            M = pd.read_csv(file_mixed, header=None).to_numpy()

            # store spectra and titles
            spectra = []
            titles = []

            if plot_clean:
                # load clean file
                file_clean = os.path.join(dir_clean, fn)
                C = pd.read_csv(file_clean, header=None).to_numpy()
                # add to lists
                spectra.append(C)
                titles.append("'%s': Clean Sound (%s)" % (fn, self._feature["feature"]))

            if plot_noise:
                # get filename of states file belonging to noise file
                fn_states = os.path.split(
                    [f for f in files_states if fn.replace(".csv", "") in f][0]
                )[-1]
                # get file idx, frame idx from states filename for noise file
                idx_file = int(fn_states.split("_")[-2])
                idx_frame = int(fn_states.split("_")[-1].split(".")[0])  # omit .csv
                # load noise file from idx_file
                file_noise = files_noise[idx_file]
                # get noise fragment from idx_frame
                N = pd.read_csv(file_noise, header=None).to_numpy()[
                    :, idx_frame : idx_frame + M.shape[1]
                ]
                # add deltas to noise fragment
                N = np.concatenate((N, librosa.feature.delta(N, mode="mirror")), axis=0)
                # add to lists
                spectra.append(N)
                titles.append("MAV Noise (%s)" % self._feature["feature"])

            # add mixed file to end of lists
            spectra.append(M)
            titles.append("'%s': Noisy Mix (%s)" % (fn, self._feature["feature"]))

            for j, Z in enumerate(spectra):
                # plot spectrum
                ax = fig.add_subplot(gs[2 * j, i])
                ph.plot_spectrum(Z[: Z.shape[0] // 2], self._feature)
                ax.set_title(titles[j])
                # plot delta-spectrum
                ax = fig.add_subplot(gs[2 * j + 1, i])
                ph.plot_spectrum(
                    Z[Z.shape[0] // 2 :], self._feature, colormap="coolwarm"
                )

    def plot_spectra_denoised(
        self,
        noise_ratio,
        categories=None,
        idx=0,
        plot_clean=True,
        plot_noise=True,
        plot_mixed=True,
        plot_predicted=True,
        enp_model_index=0,
    ):
        """Plot a selection of spectra of a denoised test set.

        Keyword arguments:
            set_name -- set to be plotted (e.g. 'Test'),
            noise_ratio -- the ratio of the noise compared to the signal,
            categories -- iterable containing the original categories
            (airplane, engine, etc.) to be plotted (default: all),
            features -- iterable containing the features to be plotted
            (default: all),
            idx -- index of the feature selected for plotting (default: 0).
            plot_clean -- whether to plot the clean spectra (default: True),
            plot_noise -- whether to plot the noise-only spectra
            (default: True)
            plot_mixed -- whether to plot the mixed-only spectra
            (default: True)
            plot_predicted: whether to plot the predicted ego-noise spectra
            (default: True)
            enp_model_index -- selects which ego-noise predictor (enp) to use
            for the prediction, if multiple are available (default: 0),
            colorbar -- whether to plot a colorbar next to each plotted
            spectrum (default: False)
        Example usage:
            plot_spectra_denoised(1.0, ['airplane', 'helicopter'], ['Stft'],
                                  plot_noise=False, plot_predicted=False)
            This plots the clean, mixed and denoised spectra of the first
            spectrogram belonging to the airplane and helicopter categories
            within the test set with a noise ratio of 1.00.
        """
        # define directories
        dir_denoised = os.path.join(
            self._dir_root_ac,
            "Features",
            "Denoised",
            "Spectra",
            "Ratio_%.2f" % noise_ratio,
            "Test",
        )
        if plot_clean:
            dir_clean = os.path.join(
                self._dir_root_ac, "Features", "Clean", "Spectra", "Test",
            )
        if plot_noise:
            # load noise files
            dir_noise = os.path.join(self._dir_root_enp, "Dataset", "Test", "Spectra",)
            files_noise = retrieve_files(dir_noise)
        if plot_mixed:
            dir_mixed = os.path.join(
                self._dir_root_ac,
                "Features",
                "Mixed",
                "Spectra",
                "Ratio_%.2f" % noise_ratio,
                "Test",
            )
        if plot_predicted:
            # select appropriate model
            dir_model_enp = sorted(
                os.listdir(os.path.join(self._dir_root_enp, "Models"))
            )[enp_model_index]
            dir_predicted = os.path.join(
                self._dir_root_enp,
                "Models",
                dir_model_enp,
                "Output",
                "Test",
                "Predicted",
            )
            files_predicted = retrieve_files(dir_predicted)
            # get model context for offset
            context = int(dir_model_enp.split("_")[-1][-1])
        if plot_noise or plot_predicted:
            # also load states files to recover noise indices
            dir_states = os.path.join(
                self._dir_root_ac, "Features", "Mixed", "States", "Test",
            )
            files_states = retrieve_files(dir_states)

        # plot all categories if not given
        filenames = [f for f in sorted(os.listdir(dir_mixed))]
        if categories is None:
            file_categories = [f.split("_")[0] for f in filenames]
            categories = sorted(list(set(file_categories)))

        # setup figure, subfigures
        n_categories = len(categories)
        n_variants = (
            1
            + int(plot_clean)
            + int(plot_noise)
            + int(plot_mixed)
            + int(plot_predicted)
        )
        fig = plt.figure(
            figsize=(6 * n_categories, 4 * n_variants), constrained_layout=True,
        )
        gs = fig.add_gridspec(n_variants * 2, n_categories)

        # loop through categories
        for i, cat in enumerate(categories):
            # get filename
            fn = [f for f in filenames if f.split("_")[0] == cat][idx]
            # load denoised file
            file_denoised = os.path.join(dir_denoised, fn)
            D = pd.read_csv(file_denoised, header=None).to_numpy()

            # store spectra and titles
            spectra = []
            titles = []

            if plot_clean:
                # load clean file
                file_clean = os.path.join(dir_clean, fn)
                C = pd.read_csv(file_clean, header=None).to_numpy()
                # add to lists
                spectra.append(C)
                titles.append("'%s': Clean Sound (%s)" % (fn, self._feature["feature"]))

            if plot_noise or plot_predicted:
                # get filename of states file belonging to noise file
                fn_states = os.path.split(
                    [f for f in files_states if fn.replace(".csv", "") in f][0]
                )[-1]
                # get file idx, frame idx from states filename for noise file
                idx_file = int(fn_states.split("_")[-2])
                idx_frame = int(fn_states.split("_")[-1].split(".")[0])

            if plot_noise:
                # load noise file from idx_file
                file_noise = files_noise[idx_file]
                # get noise fragment from idx_frame
                N = pd.read_csv(file_noise, header=None).to_numpy()[
                    :, idx_frame : idx_frame + D.shape[1]
                ]
                # add deltas to noise fragment
                N = np.concatenate((N, librosa.feature.delta(N, mode="mirror")), axis=0)
                # add to lists
                spectra.append(N)
                titles.append("MAV Noise (%s)" % self._feature["feature"])

            if plot_mixed:
                # load mixed file
                file_mixed = os.path.join(dir_mixed, fn)
                M = pd.read_csv(file_mixed, header=None).to_numpy()
                # add to lists
                spectra.append(M)
                titles.append("'%s': Noisy Mix (%s)" % (fn, self._feature["feature"]))

            if plot_predicted:
                # load predicted file
                file_pred = files_predicted[idx_file]
                P = pd.read_csv(file_pred, header=None).to_numpy()[
                    :, idx_frame - context : idx_frame - context + D.shape[1],
                ]
                # add deltas to predicted fragment
                P = np.concatenate((P, librosa.feature.delta(P, mode="mirror")), axis=0)
                spectra.append(P)
                titles.append("Predicted MAV Noise (%s)" % self._feature["feature"])

            # add denoised file to end of lists
            spectra.append(D)
            titles.append("'%s': Denoised Sound (%s)" % (fn, self._feature["feature"]))

            for j, Z in enumerate(spectra):
                # plot spectrum
                ax = fig.add_subplot(gs[2 * j, i])
                ph.plot_spectrum(Z[: Z.shape[0] // 2], self._feature)
                ax.set_title(titles[j])
                # plot delta-spectrum
                ax = fig.add_subplot(gs[2 * j + 1, i])
                ph.plot_spectrum(
                    Z[Z.shape[0] // 2 :], self._feature, colormap="coolwarm"
                )
