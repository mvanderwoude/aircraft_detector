"""
This module does stuff...
"""

# import sys
import builtins
import datetime
import queue
import os
import csv
import socket
import threading
import numpy as np
import re
import time

import paramiko
import matplotlib.pyplot as plt

import aircraft_detector.utils.plot_helper as ph


class DataAcquisition:
    def __init__(self, root_directory, server_ip, tcp_client_ip):
        # output directory of data
        self._dir_root = root_directory
        # server settings
        self._server_ip = server_ip
        self._tcp_server_port = 4999  # required by Rpi3
        self._tcp_client_ip = tcp_client_ip
        self._udp_server_port = 4245  # required by Bebop2
        # default parameters for recording, transmission
        self._mic_sample_rate = 44100  # 44100 or 48000 for usb mic
        self._mic_blocking_size = 512  # appears to be ideal
        self._mav_sample_rate = 100  # fixed in paparazzi
        self._mav_total_states = 23  # rpm4, cmd4, pos3, vel3, acc3,
        # theta3, rate3
        # queues for receiving data
        self._q_mav = queue.Queue()
        self._q_mic = queue.Queue()
        # these check if transmission has stopped
        self._mav_stopped = True
        self._mic_stopped = True
        # print y/n
        self.verbose = True

    def _print_fancy(self, *args, **kwargs):
        """Fancy print function showing timestamps"""
        if self.verbose:
            builtins.print(
                "GCS @ ",
                datetime.datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
                ": ",
                *args,
                **kwargs
            )

    def collect_data(self, tcp_client_username="pi", tcp_client_password="RaspTu"):
        """Collect a new pair of measurements (states and noise).

        Keyword arguments:
            tcp_client_username -- username of the TCP client
            (default: 'pi' for the RPi3)
            tcp_client_password -- pass word of the TCP client
            (default: 'RaspTu' for the RPi3)
        """
        # set output directories and make them if they do not exist
        dir_states = os.path.join(self._dir_root, "Raw Measurements", "Mav", "States")
        if not os.path.exists(dir_states):
            os.makedirs(dir_states)
        dir_audio = os.path.join(self._dir_root, "Raw Measurements", "Mav", "Audio")
        if not os.path.exists(dir_audio):
            os.makedirs(dir_audio)
        # generate a unique filename from the current time in UTC
        filename = datetime.datetime.utcnow().strftime("TEST_%Y-%m-%d_%H-%M-%S")
        fp_states = os.path.join(dir_states, filename + ".csv")
        fp_audio = os.path.join(dir_audio, filename + ".csv")

        # open csv file for audio logging
        with open(fp_audio, "x", newline="") as file_audio:
            writer_mic = csv.writer(file_audio)
            # open csv file for states logging
            with open(fp_states, "x", newline="") as file_states:
                writer_mav = csv.writer(file_states)
                # put states in file header
                writer_mav.writerow(
                    [
                        "delta_t",
                        "rpm_1",
                        "rpm_2",
                        "rpm_3",
                        "rpm_4",
                        "cmd_thrust",
                        "cmd_roll",
                        "cmd_pitch",
                        "cmd_yaw",
                        "pos_x",
                        "pos_y",
                        "pos_z",
                        "vel_x",
                        "vel_y",
                        "vel_z",
                        "acc_x",
                        "acc_y",
                        "acc_z",
                        "angle_phi",
                        "angle_theta",
                        "angle_psi",
                        "rate_p",
                        "rate_q",
                        "rate_r",
                    ]
                )

                # open TCP server to mic. device (RPi3)
                with socket.socket(
                    family=socket.AF_INET, type=socket.SOCK_STREAM
                ) as tcp_server:
                    tcp_server.bind((self._server_ip, self._tcp_server_port))
                    # connect to TCP client (Rpi3) via ssh
                    ssh = paramiko.SSHClient()
                    thr_ssh = threading.Thread(
                        target=self._connect_to_tcp_client,
                        args=[ssh, tcp_client_username, tcp_client_password],
                    )
                    thr_ssh.start()
                    # open TCP connection
                    self._print_fancy("Opening TCP server...")
                    tcp_server.listen(1)
                    tcp_client_socket, tcp_client = tcp_server.accept()
                    self._print_fancy("Connected to TCP client @ ", tcp_client)
                    # open the connecting socket
                    with tcp_client_socket:
                        # open UDP server to mav (bebop2)
                        with socket.socket(
                            family=socket.AF_INET, type=socket.SOCK_DGRAM
                        ) as udp_server:
                            # bind UDP client socket
                            udp_server.bind((self._server_ip, self._udp_server_port))
                            self._print_fancy(
                                "Bound UDP socket @ %s, port %d"
                                % (self._server_ip, self._udp_server_port)
                            )
                            # create threads for loggers
                            thr_tcp = threading.Thread(
                                target=self._tcp_logger, args=[tcp_client_socket]
                            )
                            thr_udp = threading.Thread(
                                target=self._udp_logger, args=[udp_server]
                            )
                            # wait until mav motors started
                            self._print_fancy("Waiting for motors to start...")
                            motors_started = False
                            while not motors_started:
                                (udp_msg, _) = udp_server.recvfrom(1024)
                                motors_started = "START" in udp_msg.decode("latin-1")
                            # signal tcp client (Rpi3) to start recording
                            tcp_client_socket.sendall(b"START")
                            self._print_fancy("MOTORS STARTED")
                            # set transmission indicators false
                            self._mav_stopped = False
                            self._mic_stopped = False
                            # launch listener threads
                            thr_tcp.start()
                            thr_udp.start()
                            # write incoming data to files until stopped
                            self._write_to_files(writer_mic, writer_mav)
                            # join threads
                            thr_tcp.join()
                            thr_udp.join()
                            self._print_fancy("MOTORS STOPPED")
                        self._print_fancy("Closed UDP socket")
                        # let TCP client (Rpi3) wrap up after closing UDP
                        time.sleep(5)
                    self._print_fancy("Closed TCP socket")
                    # close SSH after closing TCP socket and join thread
                    ssh.close()
                    thr_ssh.join()
                    self._print_fancy("Closed SSH connection")
                self._print_fancy("Closed TCP server\n\n")
        self._print_fancy(
            "Finished writing audio and state data to:\n%s,\n%s" % (fp_audio, fp_states)
        )
        return fp_audio, fp_states

    def _write_to_files(self, writer_mic, writer_mav):
        """
        Write data in queues to files using writer objects.
        """
        # write queue data to file unless both devices stopped transmitting
        # and both queues are empty
        while not (
            self._mav_stopped
            and self._mic_stopped
            and self._q_mav.empty()
            and self._q_mic.empty()
        ):
            # write to 'audio' CSV if new data arrives from mic.
            if not self._q_mic.empty():
                # decode the bytes
                audio = np.frombuffer(self._q_mic.get(), dtype="float32")
                writer_mic.writerow(audio)
                self._q_mic.task_done()
            # write to 'states' CSV if new data arrives from mav
            if not self._q_mav.empty():
                # keep only the numbers, discard junk
                states = re.findall(
                    "-?\d*\.{0,1}\d+", self._q_mav.get().decode("latin-1")
                )[: self._mav_total_states + 1]
                writer_mav.writerow(states)
                self._q_mav.task_done()
            # print items left in queue when transmission is over
            if self._mav_stopped and self._mic_stopped:
                self._print_fancy(
                    "Items left in queue: ",
                    self._q_mav.qsize(),
                    " for MAV and ",
                    self._q_mic.qsize(),
                    " for MIC.",
                )
        # close queues
        self._q_mic.join()
        self._q_mav.join()

    def _connect_to_tcp_client(self, ssh, username, password):
        """
        Connects to the TCP client (Rpi 3) and runs 'tcp_client.py' on it
        """
        # connect with ssh
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=self._tcp_client_ip, username=username, password=password)
        self._print_fancy(
            "Set up SSH connection with %s@%s" % (username, self._tcp_client_ip)
        )
        # kill previous instance if open (otherwise mic is blocked)
        ssh.exec_command("pkill -f tcp_client.py")
        # run 'tcp_client.py' which allows the mic to stream
        command = str(
            "python3 -u tcp_client.py -i "
            + self._server_ip
            + " -p "
            + str(self._tcp_server_port)
            + " -s "
            + str(self._mic_sample_rate)
            + " -b "
            + str(self._mic_blocking_size)
        )
        self._print_fancy("Command to TCP client: ", command)
        # print output of TCP client regularly
        (stdin, stdout, stderr) = ssh.exec_command(command)
        for line in stdout:
            print(line.rstrip())

    def _tcp_logger(self, socket):
        """
        Checks for messages from TCP client (mic.) and puts them in queue
        """
        sent_stop_msg = False
        # loop until motors have stopped, mic is stopped and all data is queued
        while True:
            # send 'Stop' msg to client (mic.) if motors have stopped
            if self._mav_stopped and not sent_stop_msg:
                self._print_fancy("Stopping mic...")
                socket.sendall(b"STOP")
                sent_stop_msg = True
            # check for new data
            tcp_msg = socket.recv(int(self._mic_blocking_size))
            if tcp_msg:
                # put data in queue
                self._q_mic.put(tcp_msg)
            else:
                # check if motors have stopped
                if self._mav_stopped:
                    self._print_fancy("All mic. data received.")
                    break
        # update the mic status for the main thread (collect_data)
        self._mic_stopped = True

    def _udp_logger(self, socket):
        """
        Checks for messages from UDP client (mav) and puts them in queue
        """
        # Loop until motors have stopped again ('stop' signal from MAV)
        while True:
            # check for new data
            (udp_msg, _) = socket.recvfrom(1024)
            if udp_msg:
                # put data in queue, unless 'STOP' signal is received
                if "STOP" in udp_msg.decode("latin-1"):
                    break
                self._q_mav.put(udp_msg)
        # update the mav status for the main thread (collect_data)
        self._mav_stopped = True
        self._print_fancy("BEBOP2 MOTORS STOPPED")

    def plot_data(self, file_audio, file_states, states_to_plot=None):
        """
        Plot the previously generated pair of data (audio, states)

        Keyword arguments:
            file_audio -- filepath to the audio file
            file_states -- filepath to the states file
            states_to_plot -- list of states to plot (default: all)
        """
        if states_to_plot is None:
            states_to_plot = ["rpm", "cmd", "pos", "vel", "acc", "angles", "rates"]
        n_plots = 1 + len(states_to_plot)
        fig = plt.figure(figsize=(8, 4 * n_plots))
        # plot audio
        plt.subplot(n_plots, 1, 1)
        ph.plot_audio(file_audio, self._mic_sample_rate)
        # plot states
        for i, state in enumerate(states_to_plot):
            plt.subplot(n_plots, 1, i + 2)
            ph.plot_states_raw(file_states, state)
        fig.tight_layout()
