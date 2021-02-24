import numpy as np
import librosa


def _get_fft_basis(sr, n_fft, n_bins):
    """Create a Filterbank matrix to combine FFT bins into new FFT bins

    Parameters
    ----------
    sr        : number > 0 [scalar]
        sampling rate of the incoming signal

    n_fft     : int > 0 [scalar]
        number of FFT components

    n_bins    : int > 0 [scalar]
        number of bands to generate
    """

    # Initialize the weights
    weights = np.zeros((n_bins, int(1 + n_fft // 2)))

    # get old center freqs
    f_old = librosa.filters.fft_frequencies(sr, n_fft=n_fft)

    # get new center frequencies (+ 2 padding)
    f_new = librosa.filters.fft_frequencies(sr, n_fft=2 * (n_bins - 1 + 2))

    f_diff = np.diff(f_new)
    ramps = np.subtract.outer(f_new, f_old)

    for i in range(n_bins):
        # lower and upper slopes for all bins
        lower = -ramps[i] / f_diff[i]
        upper = ramps[i + 2] / f_diff[i + 1]

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    return weights


def _downsample_fft(Z, sr, n_fft, n_bins):
    basis = _get_fft_basis(sr, n_fft, n_bins)

    return np.dot(basis, Z)


def _upsample_fft(Z, sr, n_fft):
    n_bins = Z.shape[0]

    basis = _get_fft_basis(sr, n_fft, n_bins)

    return librosa.util.nnls(basis, Z)  # kinda slow


def extract_spectrum(audio_file: str, feature_settings: dict):

    # load audio file
    y, sr = librosa.load(audio_file, sr=feature_settings["fft_sample_rate"])

    if feature_settings["feature"] == "Stft":
        # compute spectrogram
        Z = np.abs(
            librosa.stft(
                y,
                n_fft=feature_settings["stft_window_length"],
                hop_length=feature_settings["stft_hop_length"],
            )
        )

        # downsample to appropriate bins
        Z = _downsample_fft(
            Z,
            sr=sr,
            n_fft=feature_settings["stft_window_length"],
            n_bins=feature_settings["frequency_bins"],
        )

        # convert to log scale
        Z = librosa.amplitude_to_db(Z)

    elif feature_settings["feature"] == "Mel":
        # compute Mel-spectrogram
        Z = librosa.feature.melspectrogram(
            y,
            sr=sr,
            n_fft=feature_settings["stft_window_length"],
            hop_length=feature_settings["stft_hop_length"],
            n_mels=feature_settings["frequency_bins"],
        )

        # convert to log scale
        Z = librosa.power_to_db(Z)

    elif feature_settings["feature"] == "Cqt":
        # compute Constant-Q transform
        Z = np.abs(
            librosa.cqt(
                y,
                sr=sr,
                hop_length=feature_settings["stft_hop_length"],
                fmin=librosa.note_to_hz(feature_settings["cqt_min_frequency"]),
                n_bins=feature_settings["frequency_bins"],
            )
        )

        # convert to log scale
        Z = librosa.amplitude_to_db(Z)

    elif feature_settings["feature"] == "Mfcc":
        # Compute MFCCs
        Z = librosa.feature.mfcc(
            y,
            sr=sr,
            n_mfcc=feature_settings["mfc_coefficients"],
            n_fft=feature_settings["stft_window_length"],
            hop_length=feature_settings["stft_hop_length"],
            n_mels=feature_settings["frequency_bins"],
        )

    return Z


def extract_audio(
    Z, feature, params
):  # if normalized Z: unnormalize first, then pass to func.

    # convert to audio
    if feature == "Stft":
        # undo log-magnitude scaling
        S = librosa.db_to_amplitude(Z)

        # upsample
        S = _upsample_fft(S, params["fft_sample_rate"], params["stft_window_length"])

        yhat = librosa.griffinlim(S, hop_length=params["stft_hop_length"])

    elif feature == "Mel":
        # undo log-power scaling
        S = librosa.db_to_power(Z)

        yhat = librosa.feature.inverse.mel_to_audio(
            S,
            sr=params["fft_sample_rate"],
            n_fft=params["stft_window_length"],
            hop_length=params["stft_hop_length"],
        )

    elif feature == "Cqt":
        # undo log-amplitude scaling
        S = librosa.db_to_amplitude(Z)

        yhat = librosa.griffinlim_cqt(
            S,
            sr=params["fft_sample_rate"],
            hop_length=params["stft_hop_length"],
            fmin=librosa.note_to_hz(params["cqt_min_frequency"]),
        )

    elif feature == "Mfcc":

        yhat = librosa.feature.inverse.mfcc_to_audio(
            Z,
            n_mels=params["frequency_bins"],
            sr=params["fft_sample_rate"],
            n_fft=params["stft_window_length"],
            hop_length=params["stft_hop_length"],
        )

    else:
        print("Error: feature invalid")
        # throw/raise something
        return -1

    return yhat, params["fft_sample_rate"]


def normalize_spectrum(Z, znorm=False):

    if znorm:
        # z-score normalization
        Z_mean, Z_std = Z.mean(), Z.std()
        Z = (Z - Z_mean) / (Z_std + 1e-8)
        # collect statistics
        Z_stats = np.array([Z_mean, Z_std])

    else:
        # scale to (0, 1)
        Z_min, Z_max = Z.min(), Z.max()
        Z = (Z - Z_min) / (Z_max - Z_min)
        # collect statistics
        Z_stats = np.array([Z_min, Z_max])

    return Z, Z_stats


def unnormalize_spectrum(Z, Z_stats, znorm=False):

    if znorm:
        # unwrap stats
        Z_mean = Z_stats[0]
        Z_std = Z_stats[1]
        # undo znorm
        Z = Z * (Z_std - 1e-8) + Z_mean

    else:
        # unwrap stats
        Z_min = Z_stats[0]
        Z_max = Z_stats[1]
        # undo scaling
        Z = Z * (Z_max - Z_min) + Z_min

    return Z


def scale_states(states):
    # scaling constants
    RPM_SCALING = 12000
    CMD_SCALING = 12000
    HEIGHT_SCALING = 3.5
    VEL_HOR_SCALING = np.sqrt(8)  # sqrt(2**2+2**2)
    VEL_VER_SCALING = 2.0
    ACC_HOR_SCALING = np.sqrt(800)  # sqrt(20**2+20**2)
    ACC_VER_SCALING = 20.0
    ANGLE_SCALING = np.pi
    RATE_SCALING = np.pi

    states["rpm_1"] /= RPM_SCALING
    states["rpm_2"] /= RPM_SCALING
    states["rpm_3"] /= RPM_SCALING
    states["rpm_4"] /= RPM_SCALING

    states["rpm_1_delta"] /= RPM_SCALING
    states["rpm_2_delta"] /= RPM_SCALING
    states["rpm_3_delta"] /= RPM_SCALING
    states["rpm_4_delta"] /= RPM_SCALING

    states["cmd_thrust"] /= CMD_SCALING
    states["cmd_roll"] /= CMD_SCALING
    states["cmd_pitch"] /= CMD_SCALING
    states["cmd_yaw"] /= CMD_SCALING

    states["cmd_thrust_delta"] /= CMD_SCALING
    states["cmd_roll_delta"] /= CMD_SCALING
    states["cmd_pitch_delta"] /= CMD_SCALING
    states["cmd_yaw_delta"] /= CMD_SCALING

    states["height"] /= HEIGHT_SCALING

    states["vel_hor"] /= VEL_HOR_SCALING
    states["vel_ver"] /= VEL_VER_SCALING

    states["acc_hor"] /= ACC_HOR_SCALING
    states["acc_ver"] /= ACC_VER_SCALING

    states["angle_phi"] /= ANGLE_SCALING
    states["angle_theta"] /= ANGLE_SCALING
    states["angle_psi"] /= ANGLE_SCALING

    states["rate_p"] /= RATE_SCALING
    states["rate_q"] /= RATE_SCALING
    states["rate_r"] /= RATE_SCALING

    return states


def extract_relevant_states(states, input_states):
    # states = (TxS)!!!
    cols = []
    if "rpm" in input_states:
        cols.extend([0, 1, 2, 3])
    if "rpm_delta" in input_states:
        cols.extend([4, 5, 6, 7])
    if "cmd" in input_states:
        cols.extend([8, 9, 10, 11])
    if "cmd_delta" in input_states:
        cols.extend([12, 13, 14, 15])
    if "height" in input_states:
        cols.extend([16])
    if "vel" in input_states:
        cols.extend([17, 18])
    if "acc" in input_states:
        cols.extend([19, 20])
    if "angles" in input_states:
        cols.extend([21, 22, 23])
    if "rates" in input_states:
        cols.extend([24, 25, 26])

    return states[:, cols]
