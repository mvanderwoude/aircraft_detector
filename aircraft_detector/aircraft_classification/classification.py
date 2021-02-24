import os
from functools import partial
from multiprocessing import Pool

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import librosa
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

import aircraft_detector.utils.feature_helper as fh
import aircraft_detector.utils.plot_helper as ph
from aircraft_detector.utils.utils import (
    retrieve_files,
    get_feature_directory_name,
    refresh_directory,
    print_verbose,
    load_spectrum_settings,
    load_state_settings,
)
from aircraft_detector.utils.dynamic_net import (
    set_net_configuration,
    train_network,
    test_network,
    _create_network,
)


def load_feature_settings(settings_dict):
    settings = settings_dict.copy()
    if "segment_frames" not in settings:
        settings["segment_frames"] = 60
    if "segment_hop" not in settings:
        settings["segment_hop"] = 30
    if "use_delta" not in settings:
        settings["use_delta"] = True
    if "frequency_smoothing" not in settings:
        settings["frequency_smoothing"] = True
    return settings


def load_classification_settings(settings_dict):
    settings = settings_dict.copy()
    if "binary" not in settings:
        settings["binary"] = True
    if settings["binary"] is True and "aircraft" not in settings:
        settings["aircraft"] = ["airplane", "helicopter"]
    if settings["binary"] is True and "nonaircraft" not in settings:
        settings["nonaircraft"] = ["engine", "train", "wind"]
    if "categories" not in settings:
        if settings["binary"] is True:
            settings["categories"] = settings["aircraft"] + settings["nonaircraft"]
        else:
            settings["categories"] = [
                "airplane",
                "engine",
                "helicopter",
                "train",
                "wind",
            ]
    if settings["binary"] is True and "balanced_split" not in settings:
        settings["balanced_split"] = True
    if settings["balanced_split"] is True and "balance_ratios" not in settings:
        if len(settings["aircraft"]) >= len(settings["nonaircraft"]):
            settings["balance_ratios"] = [1] * len(settings["aircraft"])
        else:
            settings["balance_ratios"] = [1] * len(settings["nonaircraft"])

    return settings


class AircraftClassifier:
    def __init__(
        self,
        root_directory,
        spectrum_settings=None,
        feature_settings=None,
        classification_settings=None,
        states_settings=None,
        implicit_denoise=False,
    ):
        # set root directory
        self._dir_root = root_directory
        # set the missing spectrum settings to their defaults
        if spectrum_settings is None:
            spectrum_settings = {}
        self._spectrum = load_spectrum_settings(spectrum_settings)
        # set the missing feature settings to their defaults
        if feature_settings is None:
            feature_settings = {}
        self._feature = load_feature_settings(feature_settings)
        # set the missing classification settings to their defaults
        if classification_settings is None:
            classification_settings = {}
        self._classification = load_classification_settings(classification_settings)

        if implicit_denoise:
            # set the missing states settings to their defaults
            if states_settings is None:
                states_settings = {}
            self._states = load_state_settings(states_settings)

        self._dir_root_set = os.path.join(
            self._dir_root,
            "Aircraft Classification",
            "Parameter Sets",
            get_feature_directory_name(self._spectrum),
        )
        # verbosity
        self.verbose = True
        self.super_verbose = False

        # parts of the dataset (initialized in load_datasets)
        self._train_set = None
        self._val_set = None
        self._test_set = None

        # network configuration (initialized in set_net_configuration)
        self._net_config = None

        # define loss
        if self._classification["binary"]:
            self._loss_fn = nn.BCEWithLogitsLoss()
        else:
            self._loss_fn = nn.CrossEntropyLoss()

        # train settings (supplied in train_model)
        self._train_settings = None

    def split_features(
        self, subset=None, augmentations=None, noise_set=None, noise_ratio=None
    ):
        # default 'noise' is no noise (clean)
        if noise_set is None:
            noise_set = "Clean"

        if subset is not None:
            if type(subset) == str:
                subset = [subset]
            subsets = subset
        else:
            # split 'Train', 'Val', 'Test' set if no specific subset is given
            if augmentations is not None:
                if type(augmentations) == str:
                    augmentations = [augmentations]
                # add specific augmentation(s) to default sets
                subsets = ["Train", "Val", "Test"]
                subsets += ["Train " + a for a in augmentations]
            else:
                # use all available augmentations
                if noise_set == "Clean":
                    subsets = os.listdir(
                        os.path.join(self._dir_root_set, "Features", "Clean", "Spectra")
                    )
                else:
                    subsets = os.listdir(
                        os.path.join(
                            self._dir_root_set,
                            "Features",
                            noise_set,
                            "Spectra",
                            "Ratio_%.2f" % noise_ratio,
                        )
                    )

        # root input directory spectra
        dir_root_spectra_in = os.path.join(
            self._dir_root_set, "Features", noise_set, "Spectra"
        )
        if noise_set != "Clean":
            dir_root_spectra_in = os.path.join(
                dir_root_spectra_in, "Ratio_%.2f" % noise_ratio
            )

        for subset in sorted(subsets):
            # load 5-second spectra belonging to categories
            dir_in_spectra = os.path.join(dir_root_spectra_in, subset)
            files_spectra = retrieve_files(dir_in_spectra)
            files_spectra = [
                f
                for f in files_spectra
                if os.path.split(f)[-1].split("_")[0]
                in self._classification["categories"]
            ]

            # set output directory (augmentations i.e. 'Train Denoised' go in 'Train')
            dir_out_spectra = os.path.join(
                self._dir_root_set, "Dataset", subset.split(" ")[0], "Spectra"
            )
            # refresh directories only for non-augmented sets
            if subset in ["Train", "Val", "Test"]:
                refresh_directory(dir_out_spectra)

            # split spectra
            part = partial(self._split_spectra, dir_output=dir_out_spectra)
            with Pool(processes=os.cpu_count() - 1) as pool:
                pool.map(part, files_spectra)

            # split the states in case of implicit denoising
            if hasattr(self, "_states"):
                # load states
                dir_in_states = os.path.join(
                    self._dir_root_set, "Features", "Mixed", "States", subset
                )
                files_states = retrieve_files(dir_in_states)
                files_states = [
                    f
                    for f in files_states
                    if os.path.split(f)[-1].split("_")[0]
                    in self._classification["categories"]
                ]

                # refresh output directory only for non-augmented sets

                dir_out_states = os.path.join(
                    self._dir_root_set, "Dataset", subset.split(" ")[0], "States"
                )
                if subset in ["Train", "Val", "Test"]:
                    refresh_directory(dir_out_states)

                # split states
                part = partial(self._split_states, dir_output=dir_out_states)
                with Pool(processes=os.cpu_count() - 1) as pool:
                    pool.map(part, files_states)

            print_verbose(
                self.verbose,
                "Split %d files (%d categories) into %d files"
                % (
                    len(files_spectra),
                    len(self._classification["categories"]),
                    len(os.listdir(dir_out_spectra)),
                ),
            )

    def _split_spectra(self, file, dir_output):
        # read spectrum
        Z = pd.read_csv(file, header=None).to_numpy()
        # get indices for splitting, including offset
        offset_idx = int(
            0.5
            * (
                (Z.shape[1] - self._feature["segment_frames"])
                % self._feature["segment_hop"]
            )
        )
        split_indices = offset_idx + np.array(
            range(
                0,
                Z.shape[1] - self._feature["segment_frames"] + 1,
                self._feature["segment_hop"],
            )
        )

        for i, idx in enumerate(split_indices):
            # split spectrum (leave out delta)
            Z_s = Z[: Z.shape[0] // 2, idx : idx + self._feature["segment_frames"]]
            # smooth spectrum with savgol filter
            if self._feature["frequency_smoothing"]:
                Z_s = savgol_filter(Z_s, window_length=3, polyorder=1, axis=0)
            # scale the split back to 0-1, unless denoising implicitly
            if not hasattr(self, "_states"):
                Z_s, _ = fh.normalize_spectrum(Z_s)
            # recompute and add delta
            dZ = librosa.feature.delta(Z_s, width=9, mode="mirror")
            Z_s = np.concatenate((Z_s, dZ), axis=0)

            # export spectrum
            fn = "%s_%02d.csv" % (os.path.split(file)[-1].split(".")[0], i + 1)
            pd.DataFrame(Z_s).to_csv(
                os.path.join(dir_output, fn), header=False, index=False
            )

    def _split_states(self, file, dir_output):
        # read states
        S = pd.read_csv(file, header=None).to_numpy()
        # get indices for splitting, including offset
        offset_idx = int(
            0.5
            * (
                (S.shape[1] - self._feature["segment_frames"])
                % self._feature["segment_hop"]
            )
        )
        split_indices = offset_idx + np.array(
            range(
                0,
                S.shape[1] - self._feature["segment_frames"] + 1,
                self._feature["segment_hop"],
            )
        )

        for i, idx in enumerate(split_indices):
            # split states
            S_s = S[:, idx : idx + self._feature["segment_frames"]]
            # export states
            fn = "%s_%02d.csv" % (os.path.split(file)[-1].split(".")[0], i + 1)
            pd.DataFrame(S_s).to_csv(
                os.path.join(dir_output, fn), header=False, index=False
            )

    def load_datasets(self):

        datasets = []
        dataframes = []
        for subset in ["Train", "Val", "Test"]:
            # load spectra
            dir_spectra = os.path.join(self._dir_root_set, "Dataset", subset, "Spectra")
            files_spectra = retrieve_files(dir_spectra)

            if hasattr(self, "_states"):
                # load states
                dir_states = os.path.join(
                    self._dir_root_set, "Dataset", subset, "States"
                )
                files_states = retrieve_files(dir_states)

            # load data into torch set, dump info into df
            if not hasattr(self, "_states"):
                dataset, df = self._load_dataset(files_spectra)
            else:
                dataset, df = self._load_dataset(files_spectra, files_states)
            datasets.append(dataset)
            df.insert(0, "Set", subset)
            dataframes.append(df)

        self._train_set, self._val_set, self._test_set = datasets
        dataframe = pd.concat(dataframes)

        return dataframe

    def _load_dataset(self, files_spectra, files_states=None):

        if type(files_spectra) != np.ndarray:
            files_spectra = np.asanyarray(files_spectra)
        # list the filenames and categories for convenience (added to df later)
        filenames = np.array([os.path.split(f)[-1] for f in files_spectra])
        file_categories = np.array([f.split("_")[0] for f in filenames])

        if self._classification["binary"]:
            # get the file indices for 'aircraft' and 'nonaircraft' categories
            idcs_ac = [
                i
                for i in range(len(filenames))
                if filenames[i].split("_")[0] in self._classification["aircraft"]
            ]
            idcs_nonac = [
                i
                for i in range(len(filenames))
                if filenames[i].split("_")[0] in self._classification["nonaircraft"]
            ]

            if self._classification["balanced_split"]:
                # ensure 50/50 split between 'aircraft' and 'nonaircraft' data
                idcs_ac, idcs_nonac = self._make_balanced_split_binary(
                    filenames, idcs_ac, idcs_nonac
                )

            # sort files spectra by category
            files_spectra_ac = files_spectra[idcs_ac]
            files_spectra_nonac = files_spectra[idcs_nonac]
            files_spectra = np.concatenate(
                (files_spectra_ac, files_spectra_nonac), axis=0
            )
            # also re-order filenames and file_categories
            filenames = filenames[idcs_ac + idcs_nonac]
            file_categories = file_categories[idcs_ac + idcs_nonac]

        else:
            # keep track of occurrences per category
            occurrences = [
                (file_categories == c).sum() for c in self._classification["categories"]
            ]
            if self._classification["balanced_split"]:
                # ensure balanced split between categories
                min_occurrences = min(occurrences)

                idcs = []
                for cat in self._classification["categories"]:
                    # get all filenames belonging to category
                    idcs_cat = [
                        i
                        for i in range(len(filenames))
                        if filenames[i].split("_")[0] == cat
                    ]
                    # sample new indices for the category
                    np.random.seed(42)
                    idcs_cat = sorted(
                        np.random.choice(idcs_cat, size=min_occurrences, replace=False)
                    )
                    idcs.append(idcs_cat)

                # convert list of lists to list
                idcs = [elem for sublist in idcs for elem in sublist]
                # slice files
                files_spectra = files_spectra[idcs]
                filenames = filenames[idcs]
                file_categories = file_categories[idcs]

        spectra = []
        for file in files_spectra:
            # read into 2D array
            Z = pd.read_csv(file, header=None).to_numpy()
            # reshape into 3D array (add channel dim.)
            if self._feature["use_delta"]:
                Z = Z.reshape([2, Z.shape[0] // 2, self._feature["segment_frames"]])
            else:
                Z = np.expand_dims(Z[: Z.shape[0] // 2], axis=0)
            spectra.append(Z)

        if hasattr(self, "_states"):
            if type(files_states) != np.ndarray:
                files_states = np.asanyarray(files_states)
            # update list of files
            if self._classification["binary"]:
                files_states = files_states[idcs_ac + idcs_nonac]
            else:
                if self._classification["balanced_split"]:
                    files_states = files_states[idcs]
            # read in states
            states = []
            for file in files_states:
                S = pd.read_csv(file, header=None).to_numpy().transpose()
                S = fh.extract_relevant_states(S, self._states["states"])
                # squeeze dims
                S = S.reshape(S.shape[0], -1)
                states.append(S)

        # convert list of spectra and labels to torch dataset
        X = torch.from_numpy(np.array(spectra)).float()
        if self._classification["binary"]:
            # extract the labels (2D for some reason)
            labels_ac = np.ones((len(idcs_ac), 1))
            labels_nonac = np.zeros((len(idcs_nonac), 1))
            labels = np.concatenate((labels_ac, labels_nonac), axis=0)
            Y = torch.from_numpy(labels).float()
        else:
            labels = np.concatenate(
                [
                    i * np.ones(occurrences[i])
                    for i in range(len(self._classification["categories"]))
                ]
            )
            Y = torch.from_numpy(labels).long()

        if not hasattr(self, "_states"):
            dataset = torch.utils.data.TensorDataset(X, Y)
        else:
            X2 = torch.from_numpy(np.array(states)).float()
            X2 = X2.view(X2.size()[0], -1)  # 3D --> 2D
            dataset = torch.utils.data.TensorDataset(X, X2, Y)

        # dump info into dataframe
        df = pd.DataFrame(files_spectra, columns=["Filepath_Spectrum"])
        if hasattr(self, "_states"):
            df["Filepath_States"] = files_states
        df["Filename"] = filenames
        df["Category"] = file_categories
        df["Label"] = labels.flatten()

        return dataset, df

    def _make_balanced_split_binary(self, filenames, idcs_ac, idcs_nonac):
        if len(idcs_ac) < len(idcs_nonac):
            # get the number of occurrences of each 'nonaircraft' category
            occurrences = [
                len([f for f in filenames if f.split("_")[0] == c])
                for c in self._classification["nonaircraft"]
            ]
            # compute the amount of allowed occurences per 'nonaircraft' category
            occurrences, _ = _calc_occurrences_per_category(
                len(idcs_ac), occurrences, self._classification["balance_ratios"],
            )

            # select new nonaircraft indices
            idcs_nonac = []
            np.random.seed(42)
            for i, cat in enumerate(self._classification["nonaircraft"]):
                # get all filenames belonging to category
                idcs_cat = [
                    i
                    for i in range(len(filenames))
                    if filenames[i].split("_")[0] == cat
                ]
                # sample new indices for the category
                idcs_cat = sorted(
                    np.random.choice(idcs_cat, size=occurrences[i], replace=False)
                )
                idcs_nonac.append(idcs_cat)
            # convert list of lists back to list
            idcs_nonac = [elem for sublist in idcs_nonac for elem in sublist]

        elif len(idcs_ac) > len(idcs_nonac):
            # get the number of occurrences of each 'aircraft' category
            occurrences = [
                len([f for f in filenames if f.split("_")[0] == c])
                for c in self._classification["aircraft"]
            ]
            # compute the amount of allowed occurences per 'aircraft' category
            occurrences, _ = _calc_occurrences_per_category(
                len(idcs_nonac), occurrences, self._classification["balance_ratios"],
            )

            # select new aircraft indices
            idcs_ac = []
            np.random.seed(42)
            for i, cat in enumerate(self._classification["aircraft"]):
                # get all filenames belonging to category
                idcs_cat = [
                    i
                    for i in range(len(filenames))
                    if filenames[i].split("_")[0] == cat
                ]
                # sample new indices for the category
                idcs_cat = sorted(
                    np.random.choice(idcs_cat, size=occurrences[i], replace=False)
                )
                idcs_ac.append(idcs_cat)
            # convert list of lists back to list
            idcs_ac = [elem for sublist in idcs_ac for elem in sublist]

        return idcs_ac, idcs_nonac

    def set_net_configuration(self, layers):

        assert (
            self._test_set is not None
        ), "Please load the data via load_datasets before setting a network configuration."

        self._net_config = set_net_configuration(layers, self._test_set)

    def train_network(self, train_settings):

        # verify that network config has been set
        assert (
            self._net_config is not None
        ), "Please set a network configuration via set_net_configuration before training the network."

        # store train settings
        self._train_settings = train_settings
        # train network
        network, loss, loss_history = train_network(
            train_settings,
            self._train_set,
            self._val_set,
            self._net_config,
            self._loss_fn,
            self.verbose,
            self.super_verbose,
        )

        return network, loss, loss_history

    def test_network(self, network):

        loss = test_network(
            network, self._test_set, self._net_config["device"], self._loss_fn
        )

        return loss

    def classify_dataset(self, model, subset, dataframe):

        assert subset in ["Train", "Val", "Test"]

        # classify the train, val or test set
        if subset == "Train":
            dataset = self._train_set
        elif subset == "Val":
            dataset = self._val_set
        elif subset == "Test":
            dataset = self._test_set

        df_in = dataframe.loc[dataframe["Set"] == subset]
        df_out = self._classify_set(model, dataset, df_in)
        return df_out

    def _classify_set(self, model, dataset, dataframe):
        # create dataloader
        if len(dataset) > 2048:
            batch_size = 2048  # cap batch size to avoid memory issues
        else:
            batch_size = len(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=batch_size, drop_last=False
        )

        predictions = []
        with torch.no_grad():
            # set in eval mode
            model.eval()
            for data in dataloader:
                # send to device and do forward pass
                x = data[0].to(self._net_config["device"])
                if len(data) == 2:
                    yhat = model(x)
                else:
                    x2 = data[1].to(self._net_config["device"])
                    yhat = model(x, x2)
                # compute prediction
                if self._classification["binary"]:
                    yhat = torch.sigmoid(yhat)
                else:
                    yhat = torch.max(yhat, dim=1)[1]
                # add to predictions
                predictions.append(yhat.cpu().numpy().squeeze())

        # add predictions to dataframe
        predictions = np.array(predictions).flatten()
        dataframe["Predicted"] = predictions
        dataframe["Predicted"] = dataframe["Predicted"].round(4)

        return dataframe

    def classify_mismatched_test_set(self, model):
        # load spectra
        dir_spectra = os.path.join(self._dir_root_set, "Dataset", "Test", "Spectra")
        files_spectra = retrieve_files(dir_spectra)

        # load data into torch set, dump info into df
        if not hasattr(self, "_states"):
            dataset, df = self._load_dataset(files_spectra)
        else:
            # load states
            dir_states = os.path.join(self._dir_root_set, "Dataset", "Test", "States")
            files_states = retrieve_files(dir_states)
            dataset, df = self._load_dataset(files_spectra, files_states)

        # predict data
        df = self._classify_set(model, dataset, df)

        return df

    def save_network(self, network, loss, overwrite=False):

        # generate output filename and directory for model and config
        network_id = "%.6f" % loss
        dir_model = os.path.join(self._dir_root_set, "Models", network_id)
        fn_model = "ac_model.pt"
        fn_config = "ac_config.json"

        # create or overwrite directory
        if os.path.exists(dir_model) and not overwrite:
            print_verbose(self.verbose, "Network already exists.")
            return dir_model
        refresh_directory(dir_model)

        # save network
        torch.save(network.state_dict(), os.path.join(dir_model, fn_model))

        # save network config and settings
        config_file = open(os.path.join(dir_model, fn_config), "w")
        if not hasattr(self, "_states"):
            json.dump(
                [
                    self._net_config,
                    self._spectrum,
                    self._feature,
                    self._classification,
                    self._train_settings,
                ],
                config_file,
            )
        else:
            json.dump(
                [
                    self._net_config,
                    self._spectrum,
                    self._feature,
                    self._states,
                    self._classification,
                    self._train_settings,
                ],
                config_file,
            )
        config_file.close()

        return dir_model

    def load_network(self, network_id=None):
        if network_id is None:
            # load first network (i.e. lowest loss) of those saved
            dir_model = os.path.join(
                self._dir_root_set,
                "Models",
                sorted(os.listdir(os.path.join(self._dir_root_set, "Models")))[0],
            )
        else:
            dir_model = os.path.join(self._dir_root_set, "Models", network_id)
        # load model configuration
        fp_config = os.path.join(dir_model, "ac_config.json")

        # load class attritbutes from config file
        settings = json.load(open(fp_config))
        assert len(settings) == 5 or len(settings) == 6
        if len(settings) == 5:
            (
                self._net_config,
                self._spectrum,
                self._feature,
                self._classification,
                self._train_settings,
            ) = settings
        else:
            (
                self._net_config,
                self._spectrum,
                self._feature,
                self._states,
                self._classification,
                self._train_settings,
            ) = settings

        # load model state
        fp_model = os.path.join(dir_model, "ac_model.pt")
        model = _create_network(self._net_config)
        model.load_state_dict(torch.load(fp_model))

        return model, dir_model

    def plot_predictions(self, df, categories=None, idx=0, plot_title="default"):
        if categories is None:
            categories = df["Category"].unique().tolist()
        n_categories = len(categories)
        n_features = 1 + int(self._feature["use_delta"])

        if self._spectrum["feature"] == "Mfcc":
            f_bins = self._spectrum["mfc_coefficients"]
        else:
            f_bins = self._spectrum["frequency_bins"]

        # plot with correct aspect ratio
        fig_width = 0.07 * n_categories * self._feature["segment_frames"]
        fig_height = 0.07 * n_features * f_bins  # keep evenly sized
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
        gs = fig.add_gridspec(1, n_categories)

        for i, cat in enumerate(categories):
            gs_in = gs[0, i].subgridspec(n_features, 1)

            # get file to be plotted
            df_cat = df[df["Category"] == cat]

            # load file
            file_plot = df_cat["Filepath_Spectrum"].iloc[idx]
            Z = pd.read_csv(file_plot, header=None).to_numpy()

            # plot spectrum, add to grid
            ax = fig.add_subplot(gs_in[0])
            ph.plot_spectrum(Z[: Z.shape[0] // 2], self._spectrum)
            # set title

            if plot_title == "default":
                if self._spectrum["feature"] == "Stft":
                    feat_name = "Spectrogram"
                elif self._spectrum["feature"] == "Mel":
                    feat_name = "Mel spectrogram"
                elif self._spectrum["feature"] == "Cqt":
                    feat_name = "Constant-Q spectrogram"
                else:
                    feat_name = "MFCCs"
                title = "%s (%d bins) of class '%s'" % (feat_name, f_bins, cat)
                ax.set_title(title)

            elif plot_title == "prediction":
                fn = os.path.split(file_plot)[-1]
                pred = df_cat["Predicted"].iloc[idx]
                title = "'%s' (predicted: %.3f)" % (fn.split(".")[0], pred)
                ax.set_title(title)

            if self._feature["use_delta"]:
                # plot delta-spectrum
                ax = fig.add_subplot(gs_in[1])
                ph.plot_spectrum(
                    Z[Z.shape[0] // 2 :], self._spectrum, colormap="coolwarm"
                )
                # if j > 0:
                # ax.get_yaxis().set_visible(False)

        # fig.tight_layout()
        plt.show()
        return fig

    def print_accuracy(self, df):

        # print accuracy per category (clip-wise)
        for c in df["Category"].unique():
            df_c = df[df["Category"] == c]
            correct = np.where(df_c["Label"] == round(df_c["Predicted"]), 1, 0).sum()
            print(
                "Accuracy for category '%s' (clip): %.2f%% (%d/%d), avg. score = %.2f"
                % (
                    c,
                    correct / len(df_c) * 100,
                    correct,
                    len(df_c),
                    np.mean(df_c["Predicted"].values),
                )
            )

        if self._classification["binary"]:
            # print accuracy per label
            for c in df["Label"].unique():
                df_c = df[df["Label"] == c]
                correct = np.where(
                    df_c["Label"] == round(df_c["Predicted"]), 1, 0
                ).sum()
                label_name = "aircraft" if c == 1 else "non-aircraft"
                print(
                    "Accuracy for label '%s' (clip): %.2f%% (%d/%d), avg. score = %.2f"
                    % (
                        label_name,
                        correct / len(df_c) * 100,
                        correct,
                        len(df_c),
                        np.mean(df_c["Predicted"].values),
                    )
                )

        n_correct = np.where(df["Label"] == round(df["Predicted"]), 1, 0).sum()
        acc_clip = n_correct / len(df) * 100
        print("Accuracy: %.2f%% (%d/%d)" % (acc_clip, n_correct, len(df)))
        # print("F1-score: ", f1_score(df['Label'], round(df['Predicted'])))

        # print accuracy (recording-wise)
        rec_correct = 0
        rec_total = 0
        for c in df["Category"].unique():
            # get clips belonging to category
            df_cat = df[df["Category"] == c]
            clips = df_cat["Filepath_Spectrum"].to_numpy()

            # get recordings belonging to category
            recordings = np.unique(
                ["_".join(os.path.split(clip)[-1].split("_")[:2]) for clip in clips]
            )
            cat_total = len(recordings)

            # check how many recordings are correct and log avg. score
            scores = []
            correct = 0
            for rec in recordings:
                mask = [rec in clip for clip in clips]
                predicted = np.mean(
                    df_cat["Predicted"][mask]
                )  # avg. of each clip score
                scores.append(predicted)
                label = df_cat["Label"][mask].iloc[0]
                correct += int(round(predicted) == label)

            # update scores
            rec_total += cat_total
            rec_correct += correct

            print(
                "Accuracy for category '%s' (recording): %.2f%% (%d/%d), avg. score = %.2f"
                % (c, correct / cat_total * 100, correct, cat_total, np.mean(scores))
            )

        if self._classification["binary"]:
            # print accuracy per label

            for c in df["Label"].unique():
                label_name = "aircraft" if c == 1 else "non-aircraft"

                # get clips belonging to label
                df_cat = df[df["Label"] == c]
                clips = df_cat["Filepath_Spectrum"].to_numpy()

                # get recordings belonging to category
                recordings = np.unique(
                    ["_".join(os.path.split(clip)[-1].split("_")[:2]) for clip in clips]
                )
                cat_total = len(recordings)

                # check how many recordings are correct and log avg. score
                scores = []
                correct = 0
                for rec in recordings:
                    mask = [rec in clip for clip in clips]
                    predicted = np.mean(
                        df_cat["Predicted"][mask]
                    )  # avg. of each clip score
                    scores.append(predicted)
                    label = df_cat["Label"][mask].iloc[0]
                    correct += int(round(predicted) == label)

                print(
                    "Accuracy for label '%s' (recording): %.2f%% (%d/%d), avg. score = %.2f"
                    % (
                        label_name,
                        correct / cat_total * 100,
                        correct,
                        cat_total,
                        np.mean(scores),
                    )
                )

        acc_rec = rec_correct / rec_total * 100
        print(
            "Overall accuracy (recordings): %.2f%% (%d/%d)\n"
            % (acc_rec, rec_correct, rec_total)
        )

        return acc_clip, acc_rec

    def log_accuracy(self, df, df_log=None, index_name="Clean"):

        if df_log is None:
            df_log = pd.DataFrame(
                columns=["overall"] + self._classification["categories"]
            )

        if self._classification["binary"]:
            # total accuracy (clip)
            n_correct = np.where(df["Label"] == round(df["Predicted"]), 1, 0).sum()
            clip_acc = n_correct / len(df) * 100

            categories = df["Category"].unique()
            clip_accs = []
            rec_accs = []
            # log per category (clip)
            for c in categories:
                df_c = df[df["Category"] == c]
                # accuracy per clip
                correct_clip = np.where(
                    df_c["Label"] == round(df_c["Predicted"]), 1, 0
                ).sum()
                clip_accs.append(correct_clip / len(df_c) * 100)

                # accuracy per recording
                clips = df_c["Filepath_Spectrum"].to_numpy()
                # get recordings belonging to category
                recordings = np.unique(
                    ["_".join(os.path.split(clip)[-1].split("_")[:2]) for clip in clips]
                )
                # check how many recordings are correct
                correct_rec = 0
                for rec in recordings:
                    mask = [rec in clip for clip in clips]
                    label = df_c["Label"][mask].iloc[0]
                    predicted = np.mean(df_c["Predicted"][mask])
                    correct_rec += int(round(predicted) == label)
                rec_accs.append(correct_rec / len(recordings) * 100)

            rec_acc = sum(rec_accs) / 5

            acc_list = []
            acc_list.append("%.3g%% (%.3g%%)" % (clip_acc, rec_acc))

            for clip, rec in list(zip(clip_accs, rec_accs)):
                if clip == rec:
                    acc_list.append("%.3g%%" % clip)
                else:
                    acc_list.append("%.3g%% (%.3g%%)" % (clip, rec))
        else:
            # add this
            acc_list = []

        accuracies = pd.Series(acc_list, index=df_log.columns, name=index_name)
        df_log = df_log.append(accuracies)

        return df_log


def _calc_occurrences_per_category(n_occurrences, category_occurrences, weights):
    """Returns occurrences and new weights from number of occurrences,
        occurrence breakdown and weights.

    Example: f(450, [300, 300, 300], [0.5, 0.25, 0.25])
        --> ([225, 113, 112], [0.5, 0.25, 0.25])
    """
    # convert list to array
    if type(category_occurrences) != np.ndarray:
        category_occurrences = np.asanyarray(category_occurrences)
    if type(weights) != np.ndarray:
        weights = np.asanyarray(weights).astype(float)
    assert category_occurrences.sum() >= n_occurrences
    # weights must sum to 1
    if weights.sum() != 1.0:
        weights /= weights.sum()

    # check if simply multiplying n by weights causes overflow
    if any(n_occurrences * weights > category_occurrences):
        # check which categories overflow
        cats_overflow = np.argwhere(
            n_occurrences * weights > category_occurrences
        ).flatten()

        for i in cats_overflow:
            # compute a new balance weight
            new_weight = category_occurrences[i] / n_occurrences
            # compute the residual (overflow) weight
            o_weight = weights[i] - new_weight
            # check which categories have space for new indices
            cats_available = np.argwhere(n_occurrences * weights < category_occurrences)

            for j in cats_available:
                # increment the available weights evenly
                weights[j] += o_weight / len(cats_available)
            # update the old overflowing weights
            weights[i] = new_weight

    # calculate the number of occurrences, including whole and remainder
    occurrences = n_occurrences * weights
    occurrences_whole = occurrences.astype(int)
    occurrences_remainder = occurrences % 1

    # check for a remainde term in any of the categories
    if any(occurrences_remainder):
        n_remaining = n_occurrences - occurrences_whole.sum()
        # add to incomplete categories in descending order
        indices = np.argsort(-occurrences_remainder)
        for i in range(n_remaining):
            occurrences_whole[indices[i]] += 1

    return occurrences_whole, weights
