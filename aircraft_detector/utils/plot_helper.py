"""Module containing functions related to the plotting of data and features.

It contains the following functions:
plot_audio()
"""
import os
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def plot_audio(audio_filename, sr, waveplot=True):
    """
    thingy.

    Parameters
    ----------
    audio_filename : TYPE
        DESCRIPTION.
    sr : TYPE
        DESCRIPTION.
    waveplot : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    if audio_filename[-4:] == ".wav":
        y, _ = librosa.load(audio_filename, sr=sr)
    else:
        y = pd.read_csv(audio_filename, header=None).to_numpy().flatten()
        y = y[~np.isnan(y)]
        print(y.min(), y.max())

    if waveplot:
        librosa.display.waveplot(y, sr=sr)
    else:
        t = np.linspace(0, len(y) / sr, len(y))
        plt.plot(t, y)
        plt.xlim(0.0, t[-1])
        plt.xlabel("Time [s]")

    ylim = round(abs(y).max(), 2)
    plt.ylim([-ylim, ylim])
    # plt.ylabel('Amplitude')
    plt.title("MAV noise (sr = %.1f Hz)" % sr)
    # plt.grid()

    return


def plot_states_raw(
    states_filename, state_id, colors=["r", "g", "b", "y"]
):  # use state output from data acquisition (raw)

    df_states = pd.read_csv(states_filename)
    t = df_states["delta_t"].to_numpy()

    if state_id == "rpm":
        plt.plot(t, df_states["rpm_1"], c=colors[0], label="Rotor 1")
        plt.plot(t, df_states["rpm_2"], c=colors[1], label="Rotor 2")
        plt.plot(t, df_states["rpm_3"], c=colors[2], label="Rotor 3")
        plt.plot(t, df_states["rpm_4"], c=colors[3], label="Rotor 4")
        ylabel = "Rotor speed"
        plt.ylabel("%s [rpm]" % ylabel)

    elif state_id == "cmd":
        plt.plot(t, df_states["cmd_thrust"], c=colors[0], label="Thrust command")
        plt.plot(t, df_states["cmd_roll"], c=colors[1], label="Roll command")
        plt.plot(t, df_states["cmd_pitch"], c=colors[2], label="Pitch command")
        plt.plot(t, df_states["cmd_yaw"], c=colors[3], label="Yaw command")
        ylabel = "Stabilization commands"
        plt.ylabel(ylabel)

    elif state_id in ["pos", "position"]:
        plt.plot(t, df_states["pos_x"], c=colors[0], label="Position (x)")
        plt.plot(t, df_states["pos_y"], c=colors[1], label="Position (y)")
        plt.plot(t, df_states["pos_z"], c=colors[2], label="Position (z)")
        ylabel = "Position (NED)"
        plt.ylabel("%s [m]" % ylabel)

    elif state_id in ["vel", "velocity"]:
        plt.plot(t, df_states["vel_x"], c=colors[0], label="Velocity (x)")
        plt.plot(t, df_states["vel_y"], c=colors[1], label="Velocity (y)")
        plt.plot(t, df_states["vel_z"], c=colors[2], label="Velocity (z)")
        ylabel = "Velocity (NED)"
        plt.ylabel("%s [m/s]" % ylabel)

    elif state_id in ["acc", "acceleration"]:
        plt.plot(t, df_states["acc_x"], c=colors[0], label="Acceleration (x)")
        plt.plot(t, df_states["acc_y"], c=colors[1], label="Acceleration (y)")
        plt.plot(t, df_states["acc_z"], c=colors[2], label="Acceleration (z)")
        ylabel = "Acceleration (NED)"
        plt.ylabel("%s [m/s$^2$]" % ylabel)

    elif state_id in ["angles", "euler_angles", "attitude_angles"]:
        plt.plot(t, df_states["angle_phi"], c=colors[0], label="roll angle")
        plt.plot(t, df_states["angle_theta"], c=colors[1], label="pitch angle")
        plt.plot(t, df_states["angle_psi"], c=colors[2], label="yaw angle")
        ylabel = "Attitude"
        plt.ylabel("%s [rad]" % ylabel)

    elif state_id in ["rates", "euler_rates", "attitude_rates"]:
        plt.plot(t, df_states["rate_p"], c=colors[0], label="roll rate")
        plt.plot(t, df_states["rate_q"], c=colors[1], label="pitch rate")
        plt.plot(t, df_states["rate_r"], c=colors[2], label="yaw rate")
        ylabel = "Attitude rate"
        plt.ylabel("%s [rad/s]" % ylabel)

    else:
        print("Unknown input for 'state_id': '%s'" % state_id)
        return -1

    plt.xlim([0.0, t[-1]])
    plt.xlabel("Time [s]")
    plt.grid()
    plt.legend(loc="upper right")
    plt.title("%s over time" % ylabel)

    return


def plot_spectrum(
    Z, feature_settings, colormap="magma", colorbar=False, vrange=[None, None]
):
    """
    Hello.

    Parameters
    ----------
    Z : TYPE
        DESCRIPTION.
    feature_type : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    delta : TYPE, optional
        DESCRIPTION. The default is False.
    diff : TYPE, optional
        DESCRIPTION. The default is False.
    unnormalized : TYPE, optional
        DESCRIPTION. The default is False.
    colorbar : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    feature_type = feature_settings["feature"]
    # set frequency range based on feature
    if feature_type == "Stft":
        fmin = 0
        fmax = None
        y_axis = "linear"
    elif feature_type == "Mel":
        fmin = 0
        fmax = feature_settings["fft_sample_rate"] / 2
        y_axis = "mel"
    elif feature_type == "Cqt":
        fmin = librosa.note_to_hz(feature_settings["cqt_min_frequency"])
        fmax = None
        y_axis = "cqt_hz"
    elif feature_type == "Mfcc":
        fmin = None
        fmax = None
        y_axis = None
    else:
        pass  # throw something

    # plot spectrum
    ax = librosa.display.specshow(
        Z,
        x_axis="time",
        y_axis=y_axis,
        sr=feature_settings["fft_sample_rate"],
        hop_length=feature_settings["stft_hop_length"],
        fmin=fmin,
        fmax=fmax,
        cmap=colormap,
        vmin=vrange[0],
        vmax=vrange[1],
    )
    if feature_type == "Mfcc":
        # add ylabel instead of frequency values
        plt.ylabel("Coefficients")  # not ax.set_ylabel
    if colorbar:
        plt.colorbar(format="%.2f")

    return ax


def plot_states_synchronized(
    states, state_id, params, colors=["r", "g", "b", "y"]
):  # SxT

    t = (
        np.arange(
            0, states.shape[1] * params["stft_hop_length"], params["stft_hop_length"]
        )
        / params["fft_sample_rate"]
    )

    if state_id == "rpm":
        plt.plot(t, states[0], c=colors[0], label="Rotor 1")
        plt.plot(t, states[1], c=colors[1], label="Rotor 2")
        plt.plot(t, states[2], c=colors[2], label="Rotor 3")
        plt.plot(t, states[3], c=colors[3], label="Rotor 4")
        ylabel = "Rotor speed"

    elif state_id == "rpm_delta":
        plt.plot(t, states[4], c=colors[0], label="Rotor 1 (delta)")
        plt.plot(t, states[5], c=colors[1], label="Rotor 2 (delta)")
        plt.plot(t, states[6], c=colors[2], label="Rotor 3 (delta)")
        plt.plot(t, states[7], c=colors[3], label="Rotor 4 (delta)")
        ylabel = "Rotor speed (delta)"

    elif state_id == "cmd":
        plt.plot(t, states[8], c=colors[0], label="Thrust command")
        plt.plot(t, states[9], c=colors[1], label="Roll command")
        plt.plot(t, states[10], c=colors[2], label="Pitch command")
        plt.plot(t, states[11], c=colors[3], label="Yaw command")
        ylabel = "Stabilization commands"

    elif state_id == "cmd_delta":
        plt.plot(t, states[12], c=colors[0], label="Thrust command (delta)")
        plt.plot(t, states[13], c=colors[1], label="Roll command (delta)")
        plt.plot(t, states[14], c=colors[2], label="Pitch command (delta)")
        plt.plot(t, states[15], c=colors[3], label="Yaw command (delta)")
        ylabel = "Stabilization commands (delta)"

    elif state_id == "height":
        plt.plot(t, states[16], c=colors[0], label="Height [m]")
        ylabel = "Height"

    elif state_id in ["vel", "velocity"]:
        plt.plot(t, states[17], c=colors[0], label="Horizontal Velocity")
        plt.plot(t, states[18], c=colors[1], label="Vertical Velocity")
        ylabel = "Velocity"

    elif state_id in ["acc", "acceleration"]:
        plt.plot(t, states[19], c=colors[0], label="Horizontal Acceleration")
        plt.plot(t, states[20], c=colors[1], label="Vertical Acceleration")
        ylabel = "Acceleration"

    elif state_id in ["angles", "euler_angles", "attitude_angles"]:
        plt.plot(t, states[21], c=colors[0], label="roll angle")
        plt.plot(t, states[22], c=colors[1], label="pitch angle")
        plt.plot(t, states[23], c=colors[2], label="yaw angle")
        ylabel = "Attitude"

    elif state_id in ["rates", "euler_rates", "attitude_rates"]:
        plt.plot(t, states[24], c=colors[0], label="roll rate")
        plt.plot(t, states[25], c=colors[1], label="pitch rate")
        plt.plot(t, states[26], c=colors[2], label="yaw rate")
        ylabel = "Attitude rate"

    else:
        print("Unknown input for 'state_id': '%s'" % state_id)
        return -1

    plt.xlim([0.0, t[-1]])
    plt.xlabel("Time [s]")
    plt.grid()
    plt.legend(loc="upper right")
    plt.ylabel("%s (scaled)" % ylabel)
    plt.title("%s over time" % ylabel)

    return


def plot_spectrum_histogram(file_Z, params):

    feat = pd.read_csv(file_Z, header=None).to_numpy()
    Z = feat[: params["freq_bins"]]

    n, bins, patches = plt.hist(Z.flatten(), bins=32)

    return


def plot_delta_spectrum_histogram(file_Z, params):
    feat = pd.read_csv(file_Z, header=None).to_numpy()
    dZ = feat[params["freq_bins"] :]

    n, bins, patches = plt.hist(dZ.flatten(), bins=32)

    return


def plot_training_history(loss_history, y_max=None):
    plt.figure(figsize=(10, 10))
    plt.title("Loss History")

    # list or tuple of lists
    training_loss, validation_loss = loss_history
    epochs = np.arange(len(validation_loss)) + 1

    plt.scatter(epochs, validation_loss, label="Validation Loss")
    plt.scatter(epochs, training_loss, label="Training Loss")

    plt.xlim([1, len(epochs)])
    plt.xticks(range(0, len(epochs), 10))
    # plt.yticks(np.linspace(0, 0.7, 15))
    if y_max is None:
        y_max = max(max(training_loss), max(validation_loss))
    plt.ylim([0, 1.05 * y_max])

    plt.legend()
    plt.grid("y")
    plt.show()

    return


def plot_roc(
    labels,
    predictions,
    files=None,
    fig=None,
    title=None,
    label=None,
    color="darkorange",
):
    if fig is None:
        # prepare figure
        fig = plt.figure(figsize=(8, 4))
        plt.title(title)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid()

    if files is not None:
        # plot the recordings, instead of the segments (files)
        recordings = np.unique(
            ["_".join(os.path.split(f)[-1].split("_")[:2]) for f in files]
        )
        r_labels = []
        r_predictions = []
        for rec in recordings:
            mask = [rec in file for file in files]
            r_labels.append(labels[mask])
            r_predictions.append(np.mean(predictions[mask]))

        fpr, tpr, thresh = roc_curve(np.array(r_labels), np.array(r_predictions))
        auc = roc_auc_score(np.array(r_labels), np.array(r_predictions))

    else:
        fpr, tpr, thresh = roc_curve(labels.to_numpy(), predictions.to_numpy())
        auc = roc_auc_score(labels.to_numpy(), predictions.to_numpy())

    plt_label = " (AUC = %0.3f)" % auc
    if label is not None:
        plt_label = label + plt_label
    plt.plot(fpr, tpr, color=color, lw=2, label=plt_label)
    plt.legend(loc="lower right")

    return fig
