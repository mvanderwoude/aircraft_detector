"""Predictor.."""
import os
import shutil
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from aircraft_detector.utils.utils import (
    retrieve_files,
    get_feature_directory_name,
    refresh_directory,
    print_verbose,
    load_spectrum_settings,
    load_state_settings,
)
import aircraft_detector.utils.feature_helper as fh
import aircraft_detector.utils.pytorch_earlystopping as es
from aircraft_detector.utils.dynamic_net import Net
import aircraft_detector.utils.plot_helper as ph


from aircraft_detector.utils.dynamic_net import (
    set_net_configuration,
    train_network,
    test_network,
    _create_network,
)


class EgoNoisePredictor:
    def __init__(
        self, root_directory, spectrum_settings=None, states_settings=None,
    ):
        # set root directory
        self._dir_root = root_directory
        # set the missing feature settings to their defaults
        if spectrum_settings is None:
            spectrum_settings = {}
        self._spectrum = load_spectrum_settings(spectrum_settings)
        # set the missing states settings to their defaults
        if states_settings is None:
            states_settings = {}
        self._states = load_state_settings(states_settings)

        # derive root input directory (feature dataset) from parameters
        self._dir_root_set = os.path.join(
            self._dir_root,
            "Ego-Noise Prediction",
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

        # set loss to MSE loss
        self._loss_fn = nn.MSELoss()

        # train settings (supplied in train_model)
        self._train_settings = None

    def load_datasets(self):

        # load training, validation, test data
        self._train_set = self._load_data(
            os.path.join(self._dir_root_set, "Dataset", "Train")
        )
        self._val_set = self._load_data(
            os.path.join(self._dir_root_set, "Dataset", "Val")
        )
        self._test_set = self._load_data(
            os.path.join(self._dir_root_set, "Dataset", "Test")
        )

    def _load_data(self, dir_split):

        # load N files
        files_X = retrieve_files(os.path.join(dir_split, "States"))  # input
        files_Y = retrieve_files(os.path.join(dir_split, "Spectra"))  # output

        # load states: NxTxS
        data_X = [pd.read_csv(f, header=None).to_numpy().transpose() for f in files_X]
        # extract only relevant states
        data_X = [
            fh.extract_relevant_states(data, self._states["states"]) for data in data_X
        ]
        # load spectra: NxTxF
        data_Y = [pd.read_csv(f, header=None).to_numpy().transpose() for f in files_Y]

        if self._states["context_frames"] > 0:
            # add context to the dataset: (NxTxS, NxTxF) -> (NxT-CxCxS, NxT-CxCxF)
            data_X, data_Y = list(
                zip(*[self._add_context(dX, dY) for dX, dY in zip(data_X, data_Y)])
            )
        else:
            # add placeholder dim. for X: NxTxS -> NxTx1xS
            data_X = [np.expand_dims(X, 1) for X in data_X]

        # concatenate N and T axes to get 3D set
        data_X = np.concatenate(data_X, axis=0)
        data_Y = np.concatenate(data_Y, axis=0)
        # convert to torch dataset
        X = torch.from_numpy(data_X).float()
        Y = torch.from_numpy(data_Y).float()
        dataset = torch.utils.data.TensorDataset(X, Y)
        return dataset

    def _add_context(self, states, spectra=None):
        # 3D copy of 'original' state data: TxS -> Tx1xS
        states_extended = np.expand_dims(states, 1)

        # shift states in time dim., then add to extended array
        n = 0
        while n < self._states["context_frames"]:
            # get states at previous time index
            states_prev = np.roll(states, n + 1, axis=0)
            # add prev. states to 2nd axis of extended states
            states_extended = np.concatenate(
                (np.expand_dims(states_prev, 1), states_extended), axis=1
            )
            n += 1

        # remove first C time indices for causality: T-CxCxS
        states_extended = states_extended[self._states["context_frames"] :]

        if spectra is None:
            return states_extended
        else:
            # remove first C time indices to match length of states_extended
            spectra_modified = spectra[self._states["context_frames"] :]
            return states_extended, spectra_modified

    def set_net_configuration(self, layers):

        assert (
            self._test_set is not None
        ), "Please load the data via load_datasets before setting a network configuration."

        self._net_config = set_net_configuration(layers, self._test_set)

    def tune_hyperparameters(self, parameterization_dict, training_settings=None):
        """Use for Bayesian Optimization.

        parameterization_dict: contains ax ranges..
        """
        # convert the parameterization to the class config representation
        new_config = _convert_parameterization_to_config(parameterization_dict)
        self.set_net_configuration(new_config)
        # train, evaluate model
        _, losses, _ = self.train_network(training_settings)
        val_loss = losses[1]
        return val_loss

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

    def save_network(self, network, loss, overwrite=False):

        # generate output filename and directory for model and config
        network_id = "%.6f_c%d" % (loss, self._states["context_frames"])
        dir_model = os.path.join(self._dir_root_set, "Models", network_id)
        fn_model = "enp_model.pt"
        fn_config = "enp_config.json"

        # create or overwrite directory
        if os.path.exists(dir_model) and not overwrite:
            print_verbose(self.verbose, "Network already exists.")
            return dir_model
        refresh_directory(dir_model)

        # save network
        torch.save(network.state_dict(), os.path.join(dir_model, fn_model))

        # save network config and settings
        config_file = open(os.path.join(dir_model, fn_config), "w")
        json.dump(
            [self._net_config, self._spectrum, self._states, self._train_settings],
            config_file,
        )
        config_file.close()

        return dir_model

    def save_network_output(self, model, dir_model, subset, plot=True):

        # refresh the output directories
        output_subdirs = ["Original", "Predicted", "Residual"]
        for subdir in output_subdirs:
            refresh_directory(os.path.join(dir_model, "Output", subset, subdir))

        # load the original files (states, spectra) in the subset
        dir_states = os.path.join(self._dir_root_set, "Dataset", subset, "States")
        files_states = retrieve_files(dir_states)
        dir_spectra = os.path.join(self._dir_root_set, "Dataset", subset, "Spectra")
        files_spectra = retrieve_files(dir_spectra)

        for i in range(len(files_states)):
            # load original spectra and cut-off context
            original = pd.read_csv(files_spectra[i], header=None).to_numpy()
            if self._states["context_frames"] > 0:
                original = original[:, self._states["context_frames"] :]
            # predict spectra from states file
            predicted = self._predict(model, files_states[i], original.shape)
            # compute residual
            residual = original - predicted

            # plot if desired
            if plot:
                self._plot_model_output(original, predicted, residual)

            # save output
            fn = os.path.split(files_states[i])[-1]  # target filename
            output_spectra = [original, predicted, residual]
            for spectrum, subdir in zip(output_spectra, output_subdirs):
                # save spectrum
                dir_out = os.path.join(dir_model, "Output", subset, subdir)
                pd.DataFrame(spectrum).to_csv(
                    os.path.join(dir_out, fn), index=False, header=False
                )

    def _predict(self, network, file_states, out_shape):
        # load states
        S = pd.read_csv(file_states, header=None).to_numpy().transpose()
        S = fh.extract_relevant_states(S, self._states["states"])
        # add context or placeholder dim
        if self._states["context_frames"] > 0:
            S = self._add_context(S)
        else:
            S = np.expand_dims(S, 1)

        # predict from states
        X = torch.from_numpy(S).float().to(self._net_config["device"])
        predictions = np.zeros(out_shape)
        with torch.no_grad():
            # set in eval mode
            network.eval()
            for i, x in enumerate(X):
                # forward pass (needs fake batch dim.)
                x = x.view(1, *x.size())
                yhat = network(x)
                # store tensor (remove extra dim)
                predictions[:, i] = yhat.squeeze().cpu()

        return predictions

    def _plot_model_output(self, original, predicted, residual):

        fig = plt.figure(figsize=(6, 6))
        # plot original
        plt.subplot(3, 1, 1)
        plt.title("Original spectrum")
        ph.plot_spectrum(original, self._spectrum, colorbar=False)
        # plot predicted
        plt.subplot(3, 1, 2)
        plt.title("Predicted spectrum")
        ph.plot_spectrum(predicted, self._spectrum, colorbar=False)
        # plot residual
        plt.subplot(3, 1, 3)
        plt.title("Residual spectrum")
        ph.plot_spectrum(
            residual, self._spectrum, colormap="coolwarm", colorbar=False,
        )
        fig.tight_layout()
        plt.show()
        fig.savefig("plottyplot.eps", format="eps")

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
        fp_config = os.path.join(dir_model, "enp_config.json")
        # load class attritubtes from config file
        (
            self._net_config,
            self._spectrum,
            self._states,
            self._train_settings,
        ) = json.load(open(fp_config))

        # load model state
        fp_model = os.path.join(dir_model, "enp_model.pt")
        model = _create_network(self._net_config)
        model.load_state_dict(torch.load(fp_model))

        return model, dir_model


def _convert_parameterization_to_config(parameterization):
    """
    Example input:
    parameterization = {'LSTM_layers': 2,
        'LSTM_1_hidden_size_10': 10,
        'LSTM_1_dropout': 0.1,
        'LSTM_2_hidden_size_10': 12,
        'LSTM_3_hidden_size_10': 14,
        'Linear_layers': 2,
        'Linear_1_out_features_10': 10,
        'Dropout_1_p': 0.5,
        'Linear_2_out_features_10': 11,
        'Dropout_2_p': 0.5,
        'Linear_3_out_features_10': 12,
        'Dropout_3_p': 0.5,}

    Target output:
        layers = [
            {'layer_type': 'LSTM', 'hidden_size': 100, 'dropout': 0.1},
            {'layer_type': 'LSTM', 'hidden_size': 120},
            {'layer_type': 'Linear', 'out_features': 100},
            {'layer_type': 'Dropout', 'p': 0.5},
            {'layer_type': 'Linear', 'out_features': 110},
            {'layer_type': 'Dropout', 'p': 0.5},
            ]
    """
    # empty dictionary to store modified parameters
    parameters = {}
    # hard-coded to ensure number of dropout layers does not exceed linear layers
    parameters["Dropout_layers"] = parameterization.get("Linear_layers", 0)

    # get rid of the string indicating values should be multiplied by 10
    for key, value in parameterization.items():
        if "_10" in key:
            fixed_value = 10 * value
            fixed_key = key.replace("_10", "")
            parameters[fixed_key] = fixed_value
        else:
            parameters[key] = value

    # keep track of max. no of layers
    num_layers = {}
    layers = []
    for key, value in parameters.items():
        attributes = key.split("_")
        layer_type = attributes[0]  # i.e. 'Linear'

        if attributes[1] == "layers":
            # number of layers of this type
            num_layers[layer_type] = value
        elif int(attributes[1]) <= num_layers[layer_type]:
            # add to layers if layer number is in range
            layer_name = "_".join([attributes[0], attributes[1]])
            layer_rest = key.replace(layer_name, "")[1:]

            # merge layer dictionaries with the same keys
            layer_exists = False
            for layer in layers:
                if layer_name == layer.get("layer_type"):
                    layer_exists = True
                    break
            if layer_exists:
                layer[layer_rest] = value
            else:
                layers.append({"layer_type": layer_name, layer_rest: value})

    for layer in layers:
        # remove the trailing numbers
        layer["layer_type"] = layer["layer_type"].split("_")[0]

    return layers
