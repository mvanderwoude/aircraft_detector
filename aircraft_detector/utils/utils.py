"""Utility functions."""
import builtins
import os
import shutil


def print_verbose(verbose, *args, **kwargs):
    """
    Print the input if verbose is True.

    More readable than an if-print statement in nested loops.
    """
    if verbose:
        builtins.print(*args, **kwargs)


def refresh_directory(dir_name):
    """
    Remove and remake the specified directory.

    Parameters
    ----------
    dir_name : String
        Directory to be removed.

    Returns
    -------
    None.

    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def retrieve_files(dir_name):
    """Retrieve all files in the directory.

    Parameters
    ----------
    dir_name : str
        Directory containing files.

    Returns
    -------
    list
        List of files in the directory.

    """
    return [os.path.join(dir_name, f) for f in sorted(os.listdir(dir_name))]


def get_feature_directory_name(settings_dict):
    """Get the directory name from the supplied parameters and features."""
    dir_string = "%s_%d_%d_%d_%d" % (
        settings_dict["feature"],
        settings_dict["fft_sample_rate"],
        settings_dict["stft_window_length"],
        settings_dict["stft_hop_length"],
        settings_dict["frequency_bins"],
    )
    if settings_dict["feature"] == "cqt":
        dir_string += "_%d" % settings_dict["cqt_min_frequency"]
    if settings_dict["feature"] == "mfcc":
        dir_string += "_%d" % settings_dict["mfc_coefficients"]
    return dir_string


def load_spectrum_settings(settings_dict):
    """Set missing parameters to their defaults."""
    settings = settings_dict.copy()
    if "feature" not in settings:
        settings["feature"] = "Stft"
    if "fft_sample_rate" not in settings:
        settings["fft_sample_rate"] = 44100
    if "stft_window_length" not in settings:
        settings["stft_window_length"] = 1024
    if "stft_hop_length" not in settings:
        settings["stft_hop_length"] = settings["stft_window_length"] // 2
    if "frequency_bins" not in settings:
        settings["frequency_bins"] = 60
    if settings["feature"] == "Cqt" and "cqt_min_frequency" not in settings:
        settings["cqt_min_frequency"] = "C1"
    if settings["feature"] == "Mfcc" and "mfc_coefficients" not in settings:
        settings["mfc_coefficients"] = 13
    return settings


def load_state_settings(state_dict):
    settings = state_dict.copy()
    if "states" not in settings:
        settings["states"] = [
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
    if "context_frames" not in settings:
        settings["context_frames"] = 0
    return settings
