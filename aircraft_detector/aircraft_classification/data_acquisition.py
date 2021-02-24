"""
Uses DataAcquisition class to extract data...
"""


import os
import zipfile
import shutil
from functools import partial
from multiprocessing import Pool

import wget
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from aircraft_detector.utils.utils import (
    print_verbose,
    retrieve_files,
    refresh_directory,
)
import aircraft_detector.utils.plot_helper as ph


class DataAcquisition:
    """



    """

    def __init__(self, root_directory):
        self._dir_root = root_directory
        self._sample_rate = 44100
        self.verbose = True
        self.super_verbose = False

    def import_audio_esc50(self):
        """Download and extract the ESC-50 dataset.
        """
        # set destination
        dir_dest = os.path.join(self._dir_root, "Raw", "Aircraft")
        if not os.path.exists(dir_dest):
            os.makedirs(dir_dest)
        # download
        fp_dest = os.path.join(dir_dest, "ESC-50-master.zip")
        fp_unzipped = os.path.join(dir_dest, "ESC-50-master")
        if not os.path.exists(fp_unzipped):
            print_verbose(self.verbose, "Downloading...")
            url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
            wget.download(url, dir_dest)
            print_verbose(self.verbose, "Download finished.")
            # unzip
            with zipfile.ZipFile(fp_dest, "r") as zip_ref:
                zip_ref.extractall(dir_dest)
            print_verbose(
                self.verbose, "Extracted ESC-50 to %s" % os.path.abspath(fp_dest)
            )
        else:
            print_verbose(
                self.verbose,
                "ESC-50 has already been extracted to %s" % os.path.abspath(fp_dest),
            )

    def extract_audio_from_esc50(self, categories, db_trim=30, overwrite=False):
        """Export relevant categories from the ESC-50 dataset.

        Keyword arguments:
            categories -- iterable containing the categories from the ESC-50
            dataset that should be extracted,
            db_trim -- threshold for the trimming of silence (default: 30dB),
            overwrite -- whether to overwrite existing data (default: False),
            verbose -- whether to print each file export (default: False).
        """
        # set directories
        dir_esc50 = os.path.join(self._dir_root, "Raw", "Aircraft", "ESC-50-master")
        fp_esc50_csv = os.path.join(dir_esc50, "meta", "esc50.csv")
        dir_input = os.path.join(dir_esc50, "audio")
        dir_output = os.path.join(
            self._dir_root, "Aircraft Classification", "Audio", "Full"
        )
        # check if output directory exists
        if os.path.exists(dir_output):
            if overwrite:
                shutil.rmtree(dir_output)
            else:
                print_verbose("Output directory already exists.")
                return
        os.makedirs(os.path.join(dir_output))
        # get dataframe with filenames, categories
        df = pd.read_csv(fp_esc50_csv)
        df.drop(["fold", "target", "esc10", "src_file", "take"], axis=1, inplace=True)
        df.replace("_", "-", inplace=True, regex=True)  # less tedious later
        # extract relevant categories
        df_binary = df.loc[df["category"].isin(categories)]
        categories = df_binary["category"].unique()

        # loop over categories
        for cat in categories:
            # load files belonging to category
            files_src = df_binary.loc[df_binary["category"] == cat][
                "filename"
            ].to_list()
            # loop over files
            for i, file in enumerate(files_src):
                src = os.path.join(dir_input, file)
                # load audio
                y, sr = librosa.load(src, sr=self._sample_rate)
                # trim audio
                y_trim, _ = librosa.effects.trim(y, top_db=db_trim)
                # export audio
                fn_out = "%s_%02d.wav" % (cat, i + 1)
                dest = os.path.join(dir_output, fn_out)
                sf.write(dest, y_trim, samplerate=sr)
                # printing
                if self.super_verbose:
                    # set trim message
                    if len(y_trim) < len(y):
                        trim_msg = " (trimmed to %.3f sec.)" % (len(y_trim) / sr)
                    else:
                        trim_msg = ""
                    print("%s ---> %s%s" % (file, dest, trim_msg))

        print_verbose(
            self.verbose,
            "Finished exporting %d files (sr = %d Hz)" % (len(df_binary), sr),
        )

    def generate_silence_category(self, n_instances, duration):
        """Generates a 'silence' category consisting of white noise.

        Keyword arguments:
            n_instances -- number of instances (recordings) generated,
            duration -- duration of each recording in seconds,
            verbose -- whether to print the generation of each file
            (default: False)
        """
        # generate silence
        np.random.seed(42)
        silence = np.random.uniform(
            low=-1.0, high=1.0, size=(n_instances, duration * self._sample_rate)
        )
        # loop over instances
        for i in range(n_instances):
            # export to file
            fn = "silence_%02d.wav" % (i + 1)
            fp = os.path.join(
                self._dir_root, "Aircraft Classification", "Audio", "Full", fn
            )
            sf.write(fp, silence[i], samplerate=self._sample_rate)
            print_verbose(self.super_verbose, "Generated file: '%s'" % fp)

        print_verbose(
            self.verbose, "Finished generating %d instances of silence." % n_instances
        )

    def split_dataset(self, train_test_ratio=0.8, train_val_ratio=0.8, overwrite=False):
        """Split the dataset into a training, validation and test subset.

        Keyword arguments:
            train_test_ratio -- ratio of the training set over the complete,
            set, the remainder will be assigned to the test subset
            (default: 0.8),
            train_val_ratio -- ratio of the actual training set over the
            training set, the remainder will be assigned to the validation
            subset (default: 0.8),
            overwrite -- whether to overwrite existing data (default: False).
        """
        # directories
        dir_input = os.path.join(
            self._dir_root, "Aircraft Classification", "Audio", "Full"
        )
        dir_root_output = os.path.join(
            self._dir_root, "Aircraft Classification", "Audio"
        )
        # check if data should be overwritten if it exists
        if os.path.exists(os.path.join(dir_root_output, "Train")) and not overwrite:
            print_verbose(
                self.verbose, "Dataset already exists and should not be overwritten."
            )
            return
        # refresh the output directories
        subdirs = ["Train", "Val", "Test"]
        for subdir in subdirs:
            refresh_directory(os.path.join(dir_root_output, subdir))

        # read files into array for easy slicing
        files = np.array(retrieve_files(dir_input))
        # get categories
        file_categories = np.array([os.path.split(f)[-1].split("_")[0] for f in files])
        categories = np.unique(file_categories)
        files_per_category = len(files) // len(categories)

        # get train, val, test indices per category
        train_idcs, test_idcs = train_test_split(
            np.arange(files_per_category), train_size=train_test_ratio, random_state=42
        )
        train_idcs, val_idcs = train_test_split(
            train_idcs, train_size=train_val_ratio, random_state=42
        )
        print_verbose(
            self.verbose,
            "Split per category (Train, Val, Test): (%d, %d, %d)"
            % (len(train_idcs), len(val_idcs), len(test_idcs)),
        )

        # extract  train, val, test files using indices and export to subdirs
        indices = [train_idcs, val_idcs, test_idcs]
        for idcs, subdir in zip(indices, subdirs):
            files_set = [
                f
                for f in files
                if int(os.path.split(f)[-1].split("_")[-1].split(".")[0]) - 1 in idcs
            ]
            for file in files_set:
                dest = os.path.join(dir_root_output, subdir, os.path.split(file)[-1])
                shutil.copyfile(file, dest)

        # remove the now redundant 'Full' input directory
        shutil.rmtree(dir_input)

    def augment_training_data(self, overwrite=False):
        """Augment the training data.

        Keyword arguments:
            overwrite -- whether to overwrite existing data (default: False).
        Augmentations include:
            Pitch Shift at [-2, -1, 1, 2] octaves,
            Time Stretch with ratios of [0.70,0.85, 1.15, 1.30],
            Intra-category mixing with four random files belonging to the
            same category.
        The 'silence' category (if generated) is omitted from the augmentation.
        """
        # set directories
        dir_input = os.path.join(
            self._dir_root, "Aircraft Classification", "Audio", "Train"
        )
        dir_root_output = os.path.join(
            self._dir_root, "Aircraft Classification", "Audio"
        )

        # get files, but ignore augmentation for 'silence' category
        files = [
            os.path.join(dir_input, f)
            for f in sorted(os.listdir(dir_input))
            if os.path.split(f)[-1].split("_")[0] != "silence"
        ]

        # loop through augmentations
        augmentations = ["Pitch Shift", "Time Stretch", "Class Mix"]
        do_augmentations = []
        for aug in augmentations:
            # set output directory
            dir_output = os.path.join(dir_root_output, "Train " + aug)
            # check if it exists or should be overwritten
            if overwrite or not os.path.exists(dir_output):
                refresh_directory(dir_output)
                do_augmentations.append(aug)

        # do augmentations
        if len(do_augmentations) > 0:
            for aug in do_augmentations:
                dir_output = os.path.join(dir_root_output, "Train " + aug)
                if aug == "Class Mix":
                    # generate a list of directory-specific 'seeds' from the given seed
                    # to preserve reproducible randomness while multiprocessing
                    np.random.seed(42)
                    seeds = np.random.randint(0, 10 * len(files), len(files))

                    part = partial(
                        self._augment_class_mix, dir_out=dir_output, all_files=files
                    )

                    with Pool(processes=os.cpu_count() - 1) as pool:
                        pool.starmap(part, list(zip(files, seeds)))

                elif aug == "Pitch Shift":
                    part = partial(self._augment_pitch_shift, dir_out=dir_output)

                    with Pool(processes=os.cpu_count() - 1) as pool:
                        pool.map(part, files)

                elif aug == "Time Stretch":
                    part = partial(self._augment_time_stretch, dir_out=dir_output)

                    with Pool(processes=os.cpu_count() - 1) as pool:
                        pool.map(part, files)

            print_verbose(
                self.verbose,
                "Augmentation: %d --> %d files using %s augmentation"
                % (
                    len(files),
                    len(do_augmentations) * 4 * len(files),
                    do_augmentations,
                ),
            )
        else:
            print_verbose(self.verbose, "Augmentation has already been done.")

    def _augment_class_mix(self, file, seed, dir_out, all_files):

        # load file
        y, sr = librosa.load(file, sr=self._sample_rate)
        # mix the audio with another recording from the same category
        category = os.path.split(file)[-1].split("_")[0]
        files_category = [
            f
            for f in all_files
            if ((os.path.split(f)[-1].split("_")[0] == category) and (f != file))
        ]

        # set random, unique seed per process
        np.random.seed(seed)
        # get unique files to mix with
        n_mixes = 4
        files_to_mix = np.random.choice(files_category, size=n_mixes, replace=False)

        for i in range(n_mixes):
            # select random mix ratio and random sample offset (circular)
            mix_ratio = np.random.uniform(low=0.2, high=0.5)
            offset = int(np.random.uniform(low=0, high=5) * sr)
            # load file
            y_mix, _ = librosa.load(files_to_mix[i], sr=sr)

            # pad/truncate y_mix so that it has the same length as y
            if len(y) > len(y_mix):
                # pad with noise
                padding = np.random.uniform(
                    low=-1.0, high=1.0, size=len(y) - len(y_mix)
                )
                y_mix = np.concatenate((y_mix, padding), axis=0)
            elif len(y) < len(y_mix):
                # start sampling at random starting index
                idx = np.random.randint(len(y_mix) - len(y))
                y_mix = y_mix[idx : idx + len(y)]

            # apply offset
            y_mix = np.roll(y_mix, offset)
            # mix audio
            y_aug = y + y_mix * mix_ratio

            # export to file
            dest = os.path.join(
                dir_out, "%s-cm%d.wav" % (os.path.split(file)[-1].split(".")[0], i + 1)
            )
            sf.write(dest, y_aug, samplerate=sr)

    def _augment_pitch_shift(self, file, dir_out):
        # load file
        y, sr = librosa.load(file, sr=self._sample_rate)

        # pitch shift the audio
        shifts = [-2, -1, 1, 2]
        for i, shift in enumerate(shifts):
            y_aug = librosa.effects.pitch_shift(y, sr, shift)
            # export to file
            dest = os.path.join(
                dir_out, "%s-ps%d.wav" % (os.path.split(file)[-1].split(".")[0], i + 1),
            )
            sf.write(dest, y_aug, samplerate=sr)

    def _augment_time_stretch(self, file, dir_out):
        # load file
        y, sr = librosa.load(file, sr=self._sample_rate)

        # time stretch the audio
        stretches = [0.7, 0.85, 1.15, 1.3]
        for i, stretch in enumerate(stretches):
            y_aug = librosa.effects.time_stretch(y, stretch)
            # export to file
            dest = os.path.join(
                dir_out, "%s-ts%d.wav" % (os.path.split(file)[-1].split(".")[0], i + 1),
            )
            sf.write(dest, y_aug, samplerate=sr)

    def plot_examples(self, set_name, categories=None, idx=None):
        """Plot audio data belonging to a given dataset.

        Keyword arguments:
            set_name -- set to be plotted (e.g. 'Test')
            categories -- iterable containing the original categories
            (airplane, engine, etc.) to be plotted (default: all),
            idx -- index of the file selected for plotting (default: 0).
        """
        # get directory and available files
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
        # plot the first example if index not given
        if idx is None:
            idx = 0

        # setup figure, subfigures
        n_categories = len(categories)
        fig = plt.figure(figsize=(6 * n_categories, 2), constrained_layout=False)
        gs = fig.add_gridspec(1, n_categories)

        # loop through categories
        for i, cat in enumerate(categories):
            # load audio file
            file_audio = [
                f for f in files_audio if os.path.split(f)[-1].split("_")[0] == cat
            ][idx]
            fn = os.path.split(file_audio)[-1]
            # plot audio
            ax = fig.add_subplot(gs[i])
            ph.plot_audio(file_audio, sr=self._sample_rate, waveplot=True)
            ax.set_title("'%s' (sr = %d)" % (fn, self._sample_rate))
