"""
Created on Tue Jan 26 16:41:35 2021

@author: mark
"""
import numpy as np
import torch
import torch.nn as nn

from aircraft_detector.utils.utils import print_verbose
import aircraft_detector.utils.pytorch_earlystopping as es

"""
conv1d: in shape = (N, C_in, L_in) -> out shape = (N, C_out, L_out)
conv2d: in shape = (N, C_in, H_in, W_in) -> out_shape = (N, C_out, H_out, W_out)
linear: in_shape =  (N, L_in) -> out shape = (N, L_out)
Gru/Lstm: in_shape = L_in, out_shape = L_out
batchnorm1d: num_features = L_out
batchnorm2d: num_features = C_out
all dropout
"""


class Net(nn.Module):
    # assumes conv->fc, rec->fc, fc or conv+fc2->fc
    def __init__(
        self, config, input_shape, output_size, device=None, input_2_size=None
    ):
        super().__init__()

        # get number of layers per type
        n_conv_layers = 0
        n_rec_layers = 0
        n_fc_layers = 0
        n_fc2_layers = 0
        for layer in config:
            if "Conv" in layer["layer_type"]:
                n_conv_layers += 1
            elif layer["layer_type"] == "Linear":
                n_fc_layers += 1
            elif layer["layer_type"] == "Linear_2":
                n_fc2_layers += 1
            else:
                n_rec_layers += 1

        # set network type and generate empty network
        if n_conv_layers > 0:
            self._net_type = "CNN"
            self._modules_conv, in_features = self._generate_conv_modules(
                config, input_shape
            )
            if n_fc2_layers > 0:
                self._modules_lin2, in_features_2 = self._generate_linear_modules(
                    config, input_2_size, layer_name="Linear_2"
                )
                in_features += in_features_2
            self._modules_lin, in_features = self._generate_linear_modules(
                config, in_features
            )

        elif n_rec_layers > 0:
            self._net_type = "RNN"
            self._rnn_type = config[0]["layer_type"]
            self._device = device
            (
                self._modules_rec,
                in_features,
                self._hidden_sizes,
            ) = self._generate_recurrent_modules(config, input_shape)
            self._modules_lin, in_features = self._generate_linear_modules(
                config, in_features,
            )

        else:
            self._net_type = "MLP"
            self._modules_lin, in_features = self._generate_linear_modules(
                config, np.prod(input_shape)
            )

        # use a linear output layer
        self._out = nn.Linear(in_features, output_size)

    def _generate_conv_modules(self, config, input_shape, modules=None):
        # empty module list
        if modules is None:
            modules = nn.ModuleList()

        in_channels = input_shape[0]
        input_sizes = input_shape[1:]  # HxW
        # go over all type conv layers in config
        i = 0
        while i < len(config):
            # empty module, to be filled with 'layers' from the config
            layers_in_module = []
            # current layer settings
            layer_dict = config[i].copy()
            layer_type = layer_dict.pop("layer_type")
            # stop as soon as a fully-connected layer is found
            if layer_type == "Linear" or layer_type == "Linear_2":
                break
            # set up layer and its parameters
            layer = getattr(nn, layer_type)  # i.e. nn.Conv2d
            layer = layer(in_channels, **layer_dict)
            # add to module
            layers_in_module.append(layer)
            i += 1
            # set new 'in_channels' to current 'out_channels'
            in_channels = layer_dict["out_channels"]
            # calculate new 'input_sizes' (height, width)
            input_sizes = _calc_conv_layer_output_shape(
                layer_type, layer_dict, input_sizes
            )

            # apply batch normalization if in config and before relu
            if i < len(config):
                if (
                    "BatchNorm" in config[i]["layer_type"]
                    and config[i].get("location", None) == "before"
                ):
                    bn_dict = config[i].copy()
                    _ = bn_dict.pop("location")
                    bn_type = bn_dict.pop("layer_type")
                    bn = getattr(nn, bn_type)
                    # supply new 'in_channels' and eps, along with layer parameters
                    bn = bn(in_channels, eps=1e-8, **bn_dict)
                    layers_in_module.append(bn)
                    i += 1

            # apply pooling if in config
            if i < len(config):
                if "MaxPool" in config[i]["layer_type"]:
                    pool_dict = config[i].copy()
                    pool_type = pool_dict.pop("layer_type")
                    pool = getattr(nn, pool_type)
                    # supply parameters
                    pool = pool(**pool_dict)
                    layers_in_module.append(pool)
                    i += 1
                    # calculate new 'input_sizes' (height, width)
                    input_sizes = _calc_conv_layer_output_shape(
                        pool_type, pool_dict, input_sizes
                    )

            # add ReLU
            layers_in_module.append(nn.ReLU())

            # apply dropout if in config
            if i < len(config):
                if "Dropout" in config[i]["layer_type"]:
                    dropout_dict = config[i].copy()
                    _ = dropout_dict.pop("layer_type")
                    # supply parameters in dict
                    dropout = nn.Dropout(**dropout_dict)
                    layers_in_module.append(dropout)
                    i += 1

            # apply batch normalization if in config and after relu (default)
            if i < len(config):
                if (
                    "BatchNorm" in config[i]["layer_type"]
                    and config[i].get("location", None) != "before"
                ):
                    bn_dict = config[i].copy()
                    _ = bn_dict.pop("location")
                    bn_type = bn_dict.pop("layer_type")
                    bn = getattr(nn, bn_type)
                    # supply new 'in_channels' and eps, along with layer parameters
                    bn = bn(in_channels, eps=1e-8, **bn_dict)
                    layers_in_module.append(bn)
                    i += 1

            # add module to module list
            module = nn.Sequential(*layers_in_module)
            modules.append(module)

        # calculate number of output units (required for FC layers)
        output_units = in_channels * np.prod(input_sizes)

        return modules, output_units

    def _generate_recurrent_modules(self, config, input_shape, modules=None):
        # empty module list
        if modules is None:
            modules = nn.ModuleList()

        # hidden sizes are used in forward() to init. hidden states with 0s
        hidden_sizes = []

        input_size = input_shape[-1]  # no. of input features (freq. bins)
        # go over all recurrent layers in config
        i = 0
        while i < len(config):
            # current layer settings
            layer_dict = config[i].copy()
            layer_type = layer_dict.pop("layer_type")
            # stop as soon as a fully-connected layer is found
            if layer_type == "Linear":
                break
            # set up layer and its parameters
            layer = getattr(nn, layer_type)  # i.e. nn.Conv2d
            layer = layer(input_size, **layer_dict, batch_first=True)
            # add to module list
            modules.append(layer)
            i += 1
            # set new 'input_size' to current 'hidden_size'
            input_size = layer_dict["hidden_size"]
            hidden_sizes.append(input_size)

        return modules, input_size, hidden_sizes

    def _generate_linear_modules(
        self, config, in_features, modules=None, layer_name="Linear"
    ):
        # empty module list
        if modules is None:
            modules = nn.ModuleList()

        # skip layers until the first fully-connected layer is found
        i = 0
        while config[i]["layer_type"] != layer_name:
            i += 1
            if i >= len(config):
                # in case no linear layers in config (not recommended)
                return modules, in_features

        # search remaining layers
        while i < len(config):
            # check for interference with second linear module
            if (config[i]["layer_type"] == "Linear" and layer_name == "Linear_2") or (
                config[i]["layer_type"] == "Linear_2" and layer_name == "Linear"
            ):
                break

            # empty module, to be filled with 'layers' from the config
            layers_in_module = []
            # current layer settings
            layer_dict = config[i].copy()
            layer_type = layer_dict.pop("layer_type").split("_")[0]
            # set up layer and its parameters
            layer = getattr(nn, layer_type)  # i.e. nn.Conv2d
            layer = layer(in_features, **layer_dict)
            # add to module
            layers_in_module.append(layer)
            i += 1

            # set new 'in_features' to current 'out_features'
            in_features = layer_dict["out_features"]

            # add ReLU
            layers_in_module.append(nn.ReLU())

            # apply dropout if in config
            if i < len(config):
                if "Dropout" in config[i]["layer_type"]:
                    dropout_dict = config[i].copy()
                    _ = dropout_dict.pop("layer_type")
                    # supply parameters in dict
                    dropout = nn.Dropout(**dropout_dict)
                    layers_in_module.append(dropout)
                    i += 1

            # add module to module list
            module = nn.Sequential(*layers_in_module)
            modules.append(module)

        return modules, in_features

    def forward(self, x, x2=None):  # add support for double input net!!!
        if self._net_type == "CNN":
            x = self._forward_convolutional(x)
        elif self._net_type == "RNN":
            x = self._forward_recurrent(x)
        else:
            pass
        # reshape output for fully-connected layer
        x = x.reshape(x.size(0), -1)
        if x2 is not None:
            for module in self._modules_lin2:
                x2 = module(x2)
            x = torch.cat((x, x2), dim=1)
        # forward pass: FC layers
        for module in self._modules_lin:
            x = module(x)
        # linear output layer
        x = self._out(x)

        return x

    def _forward_convolutional(self, x):
        for module in self._modules_conv:
            x = module(x)

        return x

    def _forward_recurrent(self, x):
        for i, module in enumerate(self._modules_rec):
            # initialize hidden state with zeros
            hidden_size = self._hidden_sizes[i]
            h0 = (
                torch.zeros(1, x.size(0), hidden_size).requires_grad_().to(self._device)
            )
            # initialize hidden cell state with zeros if lstm is used
            if self._rnn_type == "LSTM":  # or check type in module??
                c0 = (
                    torch.zeros(1, x.size(0), hidden_size)
                    .requires_grad_()
                    .to(self._device)
                )
                # detach h0, c0 to avoid BPTT through previous batches
                x, _ = module(x, (h0.detach(), c0.detach()))
            else:
                # detach h0 to avoid BPTT through previous batches
                x, _ = module(x, h0.detach())

        # extract output from last timestep
        x = x[:, -1, :]

        return x


def _calc_conv_layer_output_shape(layer_type, layer_dict, input_shape):

    assert layer_type in ["Conv1d", "Conv2d", "MaxPool1d", "MaxPool2d"]

    if "1d" in layer_type:
        in_shape = input_shape[0]
        # get layer params or default
        kernel = layer_dict["kernel_size"]
        if layer_type == "Conv1d":
            stride = layer_dict.get("stride", 1)
        else:
            stride = layer_dict.get("stride", 2)
        padding = layer_dict.get("padding", 0)
        dilation = layer_dict.get("dilation", 1)
        # compute output length
        out_shape = int(
            (in_shape + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        )
    else:
        in_height = input_shape[0]
        in_width = input_shape[1]
        # get layer params or default
        kernel = layer_dict["kernel_size"]
        if layer_type == "Conv2d":
            stride = layer_dict.get("stride", (1, 1))
        else:
            stride = layer_dict.get("stride", (2, 2))
        padding = layer_dict.get("padding", (0, 0))
        dilation = layer_dict.get("dilation", (1, 1))
        # compute output shape
        out_height = int(
            (in_height + 2 * padding[0] - dilation[0] * (kernel[0] - 1) - 1) / stride[0]
            + 1
        )
        out_width = int(
            (in_width + 2 * padding[1] - dilation[1] * (kernel[1] - 1) - 1) / stride[1]
            + 1
        )
        out_shape = [out_height, out_width]

    return out_shape


# network configuration


def set_net_configuration(layers, test_set):
    # fetch input shape and output size from test set
    input_shape = list(test_set.tensors[0].size()[1:])
    output_size = int(np.prod(test_set.tensors[-1].size()[1:]))  # cast to int for json

    # set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # set network configuration
    net_config = {
        "layers": layers,
        "input_shape": input_shape,
        "output_size": output_size,
        "device": device,
    }
    if len(test_set.tensors) > 2:
        net_config["input_2_size"] = np.prod(test_set.tensors[1].size()[1:])

    return net_config


def train_network(
    train_settings,
    train_set,
    val_set,
    net_config,
    loss_fn,
    verbose=True,
    super_verbose=False,
):

    # reset the seed
    torch.manual_seed(42)

    # create network from config
    network = _create_network(net_config)

    # copy optimizer settings to avoid modifying train_settings
    dict_optimizer = train_settings["optimizer"].copy()
    # select the optimizer in torch.optim from settings
    optimizer = getattr(torch.optim, dict_optimizer.pop("optimizer"))
    # bind network, unpack optimizer settings
    optimizer = optimizer(network.parameters(), **dict_optimizer)

    if "lr_scheduler" in train_settings:
        # copy scheduler settings to avoid modifying train_settings
        dict_scheduler = train_settings["lr_scheduler"].copy()
        # select the lr scheduler in torch.optim from settings
        lr_scheduler = getattr(
            torch.optim.lr_scheduler, dict_scheduler.pop("scheduler")
        )
        # bind optimizer, unpack scheduler settings
        lr_scheduler = lr_scheduler(optimizer, **dict_scheduler)

    # create train dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=train_settings["batch_size"],
        shuffle=True,
        drop_last=False,
    )
    # create validation dataloader
    if len(val_set) > 2048:
        val_batch_size = 2048  # cap batch size to avoid memory issues
    else:
        val_batch_size = len(val_set)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set, batch_size=val_batch_size, drop_last=False
    )

    if "es_patience" in train_settings:
        # set up early stopping checkpoint
        fp_checkpoint = "checkpoint-es.pt"
        early_stopping = es.EarlyStopping(
            patience=train_settings["es_patience"],
            delta=1e-7,
            verbose=super_verbose,
            output_fp=fp_checkpoint,
        )

    training_loss_history = []
    validation_loss_history = []
    # loop over epochs
    for epoch in range(train_settings["epochs"]):
        train_losses = []
        # set in training mode
        network.train()
        for data in train_loader:
            # to device (gpu/cpu)
            x_train = data[0].to(net_config["device"])
            if len(data) > 2:
                x2_train = data[1].to(net_config["device"])
            y_train = data[-1].to(net_config["device"])
            # clear gradient of optimizer
            optimizer.zero_grad()
            # forward pass
            if len(data) == 2:
                yhat = network(x_train)
            else:
                yhat = network(x_train, x2_train)
            # compute loss
            loss = loss_fn(yhat, y_train)
            # backward pass
            loss.backward()
            # record loss
            train_losses.append(loss.item())
            # update parameters
            optimizer.step()
        # record loss and update loss history
        training_loss = np.mean(train_losses)
        training_loss_history.append(training_loss)

        # validation loss
        with torch.no_grad():
            val_losses = []
            # set in eval mode
            network.eval()
            for data in val_loader:
                # to device (gpu/cpu)
                x_val = data[0].to(net_config["device"])
                if len(data) > 2:
                    x2_val = data[1].to(net_config["device"])
                y_val = data[-1].to(net_config["device"])
                # forward pass
                if len(data) == 2:
                    yhat = network(x_val)
                else:
                    yhat = network(x_val, x2_val)
                # compute loss
                val_loss = loss_fn(yhat, y_val)
                # record loss
                val_losses.append(val_loss.item())
        # record loss and update loss history
        validation_loss = np.mean(val_losses)
        validation_loss_history.append(validation_loss)

        print_verbose(
            super_verbose,
            "epoch %d: training loss = %.6f, validation loss = %.6f"
            % (epoch + 1, training_loss, validation_loss),
        )

        if "es_patience" in train_settings:
            # check early stopping criterion
            early_stopping(validation_loss, network)
            if early_stopping.early_stop:
                # get training loss at best epoch
                training_loss = training_loss_history[
                    epoch - train_settings["es_patience"]
                ]
                # get validation loss at best epoch
                validation_loss = early_stopping.val_loss_min
                print_verbose(
                    super_verbose,
                    "Early stopping (using model at epoch %d with val. loss %.5f)"
                    % (epoch + 1 - train_settings["es_patience"], validation_loss),
                )
                # end training
                break

        if "lr_scheduler" in train_settings:
            # update learning rate
            lr_scheduler.step(validation_loss)

    if "es_patience" in train_settings:
        # load network from checkpoint
        network.load_state_dict(torch.load(early_stopping.output_fp))
        # delete checkpoint !!!

    loss = (training_loss, validation_loss)
    loss_history = (training_loss_history, validation_loss_history)
    return network, loss, loss_history


def _create_network(net_config, verbose=True, super_verbose=False):
    # set up network
    network = Net(*net_config.values())
    network.to(net_config["device"])

    print_verbose(verbose, "Device: %s." % net_config["device"])
    print_verbose(super_verbose, network)
    print_verbose(
        verbose,
        "Number of trainable parameters in network: %d."
        % sum([p.numel() for p in network.parameters()]),
    )

    return network


def test_network(network, test_set, device, loss_fn):

    # create test dataloader
    if len(test_set) > 2048:
        test_batch_size = 2048  # cap batch size to avoid memory issues
    else:
        test_batch_size = len(test_set)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=test_batch_size, drop_last=False
    )

    # calc. loss
    with torch.no_grad():
        losses = []
        # set in eval mode
        network.eval()
        for data in test_loader:
            # to device (gpu/cpu)
            x = data[0].to(device)
            if len(data) > 2:
                x2 = data[1].to(device)
            y = data[-1].to(device)
            # forward pass
            if len(data) == 2:
                yhat = network(x)
            else:
                yhat = network(x, x2)
            # compute loss
            loss = loss_fn(yhat, y)
            # record loss
            losses.append(loss.item())
    # record loss and update loss history
    loss = np.mean(losses)

    return loss
