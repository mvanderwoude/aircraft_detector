"""Module containing functions related to the plotting of data and features.

It contains the following functions:
plot_audio()
"""

import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt


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
    states_filename, state_id
):  # use state output from data acquisition (raw)

    df_states = pd.read_csv(states_filename)
    t = df_states["delta_t"].to_numpy()

    if state_id == "rpm":
        plt.plot(t, df_states["rpm_1"], c="r", label="Rotor 1")
        plt.plot(t, df_states["rpm_2"], c="g", label="Rotor 2")
        plt.plot(t, df_states["rpm_3"], c="b", label="Rotor 3")
        plt.plot(t, df_states["rpm_4"], c="y", label="Rotor 4")
        ylabel = "Rotor speed"
        plt.ylabel("%s [rpm]" % ylabel)

    elif state_id == "cmd":
        plt.plot(t, df_states["cmd_thrust"], c="r", label="Thrust command")
        plt.plot(t, df_states["cmd_roll"], c="g", label="Roll command")
        plt.plot(t, df_states["cmd_pitch"], c="b", label="Pitch command")
        plt.plot(t, df_states["cmd_yaw"], c="y", label="Yaw command")
        ylabel = "Stabilization commands"
        plt.ylabel(ylabel)

    elif state_id in ["pos", "position"]:
        plt.plot(t, df_states["pos_x"], c="r", label="Position (x)")
        plt.plot(t, df_states["pos_y"], c="g", label="Position (y)")
        plt.plot(t, df_states["pos_z"], c="b", label="Position (z)")
        ylabel = "Position (NED)"
        plt.ylabel("%s [m]" % ylabel)

    elif state_id in ["vel", "velocity"]:
        plt.plot(t, df_states["vel_x"], c="r", label="Velocity (x)")
        plt.plot(t, df_states["vel_y"], c="g", label="Velocity (y)")
        plt.plot(t, df_states["vel_z"], c="b", label="Velocity (z)")
        ylabel = "Velocity (NED)"
        plt.ylabel("%s [m/s]" % ylabel)

    elif state_id in ["acc", "acceleration"]:
        plt.plot(t, df_states["acc_x"], c="r", label="Acceleration (x)")
        plt.plot(t, df_states["acc_y"], c="g", label="Acceleration (y)")
        plt.plot(t, df_states["acc_z"], c="b", label="Acceleration (z)")
        ylabel = "Acceleration (NED)"
        plt.ylabel("%s [m/s$^2$]" % ylabel)

    elif state_id in ["angles", "euler_angles", "attitude_angles"]:
        plt.plot(t, df_states["angle_phi"], c="r", label="roll angle")
        plt.plot(t, df_states["angle_theta"], c="g", label="pitch angle")
        plt.plot(t, df_states["angle_psi"], c="b", label="yaw angle")
        ylabel = "Attitude"
        plt.ylabel("%s [rad]" % ylabel)

    elif state_id in ["rates", "euler_rates", "attitude_rates"]:
        plt.plot(t, df_states["rate_p"], c="r", label="roll rate")
        plt.plot(t, df_states["rate_q"], c="g", label="pitch rate")
        plt.plot(t, df_states["rate_r"], c="b", label="yaw rate")
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
    Z, feature_type, params, delta=False, diff=False, unnormalized=False, colorbar=False
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
    Z_bins = Z.shape[0] // 2

    if delta:
        Z = Z[Z_bins:]
    else:
        Z = Z[:Z_bins]

    if delta or diff:
        # set limits
        vlim = np.abs(Z).max()
        vrange = [-vlim, vlim]
        cmap = "coolwarm"
    elif unnormalized:
        vrange = [Z.min(), Z.max()]
        cmap = "magma"
    else:
        vrange = [0, 1]
        cmap = "magma"

    if feature_type == "Stft":
        fmin = 0
        fmax = None
        y_axis = "linear"
    elif feature_type == "Mel":
        fmin = 0
        fmax = params["fft_sample_rate"] / 2
        y_axis = "mel"
    elif feature_type == "Cqt":
        fmin = librosa.note_to_hz(params["cqt_min_frequency"])
        fmax = None
        y_axis = "cqt_hz"
    elif feature_type == "Mfcc":
        fmin = None
        fmax = None
        y_axis = None
    else:
        print("Error: feature invalid")
        return -1
        # throw something

    ax = librosa.display.specshow(
        Z,
        x_axis="time",
        y_axis=y_axis,
        sr=params["fft_sample_rate"],
        hop_length=params["stft_hop_length"],
        fmin=fmin,
        fmax=fmax,
        cmap=cmap,
        vmin=vrange[0],
        vmax=vrange[1],
    )

    if feature_type == "Mfcc":
        plt.ylabel("Coefficients")  # not ax.set_ylabel
        # ax.get_yaxis().set_visible(True)

    if colorbar:
        plt.colorbar(format="%.2f")

    return ax


def plot_states_synchronized(states, state_id, params):  # SxT

    t = (
        np.arange(
            0, states.shape[1] * params["stft_hop_length"], params["stft_hop_length"]
        )
        / params["fft_sample_rate"]
    )

    if state_id == "rpm":
        plt.plot(t, states[0], c="r", label="Rotor 1")
        plt.plot(t, states[1], c="g", label="Rotor 2")
        plt.plot(t, states[2], c="b", label="Rotor 3")
        plt.plot(t, states[3], c="y", label="Rotor 4")
        ylabel = "Rotor speed"

    elif state_id == "rpm_delta":
        plt.plot(t, states[4], c="r", label="Rotor 1 (delta)")
        plt.plot(t, states[5], c="g", label="Rotor 2 (delta)")
        plt.plot(t, states[6], c="b", label="Rotor 3 (delta)")
        plt.plot(t, states[7], c="y", label="Rotor 4 (delta)")
        ylabel = "Rotor speed (delta)"

    elif state_id == "cmd":
        plt.plot(t, states[8], c="r", label="Thrust command")
        plt.plot(t, states[9], c="g", label="Roll command")
        plt.plot(t, states[10], c="b", label="Pitch command")
        plt.plot(t, states[11], c="y", label="Yaw command")
        ylabel = "Stabilization commands"

    elif state_id == "cmd_delta":
        plt.plot(t, states[12], c="r", label="Thrust command (delta)")
        plt.plot(t, states[13], c="g", label="Roll command (delta)")
        plt.plot(t, states[14], c="b", label="Pitch command (delta)")
        plt.plot(t, states[15], c="y", label="Yaw command (delta)")
        ylabel = "Stabilization commands (delta)"

    elif state_id == "height":
        plt.plot(t, states[16], c="r", label="Height [m]")
        ylabel = "Height"

    elif state_id in ["vel", "velocity"]:
        plt.plot(t, states[17], c="r", label="Horizontal Velocity")
        plt.plot(t, states[18], c="g", label="Vertical Velocity")
        ylabel = "Velocity"

    elif state_id in ["acc", "acceleration"]:
        plt.plot(t, states[19], c="r", label="Horizontal Acceleration")
        plt.plot(t, states[20], c="g", label="Vertical Acceleration")
        ylabel = "Acceleration"

    elif state_id in ["angles", "euler_angles", "attitude_angles"]:
        plt.plot(t, states[21], c="r", label="roll angle")
        plt.plot(t, states[22], c="g", label="pitch angle")
        plt.plot(t, states[23], c="b", label="yaw angle")
        ylabel = "Attitude"

    elif state_id in ["rates", "euler_rates", "attitude_rates"]:
        # plt.plot(t, states[24], c='r', label='roll rate')
        # plt.plot(t, states[25], c='g', label='pitch rate')
        # plt.plot(t, states[26], c='b', label='yaw rate')
        plt.plot(t, states[16], c="r", label="roll rate")
        plt.plot(t, states[17], c="g", label="pitch rate")
        plt.plot(t, states[18], c="b", label="yaw rate")
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


##### old
def plot_mav_data(audio_filename, state_filename, states=["rpm"], fs=None):

    # load audio
    y, fs = librosa.load(audio_filename, sr=fs)

    # load state data
    df_states = pd.read_csv(state_filename)

    # construct time series
    t_mic = np.linspace(0, len(y) / fs, len(y))
    t_mav = df_states["deltaT"]
    t_end_max = np.max([t_mic[-1], t_mav.iloc[-1]])
    print("Recording times: ", t_mic[-1], "s (mic)", t_mav.iloc[-1], "s (mav)")

    # construct subplots (audio + clusters of states)
    i = 0
    if "all" in states:
        n = 8
    else:
        n = 1 + len(states)
    fig, axes = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(20, 5 * n))

    # plot audio
    axes[i].plot(t_mic, y)
    axes[i].set_xlim([0.0, t_end_max])
    axes[i].set_xlabel("Time [s]")
    axes[i].set_ylabel("Amplitude")
    axes[i].set_title("MAV noise  (fs = %.1f Hz)" % fs)
    axes[i].grid()

    # rpm
    if "rpm" in states or "all" in states:
        i += 1
        axes[i].plot(t_mav, df_states["rpm1"], c="r", label="rpm1")
        axes[i].plot(t_mav, df_states["rpm2"], c="g", label="rpm2")
        axes[i].plot(t_mav, df_states["rpm3"], c="b", label="rpm3")
        axes[i].plot(t_mav, df_states["rpm4"], c="y", label="rpm4")
        axes[i].set_ylim([0.0, 12000])
        axes[i].axhline(12000, c="k", ls="--", label="rpmMAX")
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Rotor RPM")
        axes[i].set_title("Rotor RPM over time")

    if "cmd" in states or "all" in states:
        i += 1
        axes[i].plot(t_mav, df_states["cmdThrust"], c="r", label="cmdThrust")
        axes[i].plot(t_mav, df_states["cmdRoll"], c="g", label="cmdRoll")
        axes[i].plot(t_mav, df_states["cmdPitch"], c="b", label="cmdPitch")
        axes[i].plot(t_mav, df_states["cmdYaw"], c="y", label="cmdYaw")
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Input Commands")
        axes[i].set_title("Input commands over time")

    if "z" in states or "all" in states:
        i += 1
        axes[i].plot(t_mav, df_states["z"], c="r", label="z")
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("MAV height [m]")
        axes[i].set_title("MAV height over time")

    if "xyzdot" in states or "all" in states:
        i += 1
        axes[i].plot(t_mav, df_states["xdot"], c="r", label="xdot")
        axes[i].plot(t_mav, df_states["ydot"], c="g", label="ydot")
        axes[i].plot(t_mav, df_states["zdot"], c="b", label="zdot")
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Body velocities [m/s]")
        axes[i].set_title("Body velocities over time")

    if "xyzdotdot" in states or "all" in states:
        i += 1
        axes[i].plot(t_mav, df_states["xdotdot"], c="r", label="xdotdot")
        axes[i].plot(t_mav, df_states["ydotdot"], c="g", label="ydotdot")
        axes[i].plot(t_mav, df_states["zdotdot"], c="b", label="zdotdot")
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Body accelerations [m/s2]")
        axes[i].set_title("Body accelerations over time")

    if "theta" in states or "all" in states:
        i += 1
        axes[i].plot(t_mav, df_states["phi"], c="r", label="phi")
        axes[i].plot(t_mav, df_states["theta"], c="g", label="theta")
        axes[i].plot(t_mav, df_states["psi"], c="b", label="psi")
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Body attitudes [rad]")
        axes[i].set_title("Body attitudes over time")

    if "thetadot" in states or "all" in states:
        i += 1
        axes[i].plot(t_mav, df_states["phidot"], c="r", label="phidot")
        axes[i].plot(t_mav, df_states["thetadot"], c="g", label="thetadot")
        axes[i].plot(t_mav, df_states["psidot"], c="b", label="psidot")
        axes[i].grid()
        axes[i].legend()
        axes[i].set_xlabel("Time [s]")
        axes[i].set_ylabel("Body attitude rates [rad/s]")
        axes[i].set_title("Body attitude rates over time")

    fig.tight_layout()
    plt.show()
    return
