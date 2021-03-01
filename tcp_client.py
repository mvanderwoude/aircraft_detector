import socket
import argparse
import sounddevice as sd
#import soundfile as sf
import queue
import time
import threading

class TcpClientAudio:
    
    def __init__(self, server_ip='192.168.0.0', server_port=10000, sample_rate=48000, block_size=4096):
        self._ip = server_ip
        self._port = server_port 
        self._sample_rate = int(sample_rate)
        self._block_size = int(block_size) # amount of samples sent per callback
        self._q = queue.Queue()
        
        self._stop_recording = False
        
        return
    '''
    def _audio_callback(self, indata, samples, time_received, status):
        # send audio data to server
        #sock.sendall(indata.copy())
        self._q.put(indata[::self._down_sample_ratio])
        #self._q.put(indata.copy())
        #t = time_received.inputBufferAdcTime
        #print(t)
        print("Sent ", samples, " audio samples: time = ", datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
              ", dt = ", time.time() - self._starting_time) 
        
        return
    '''
    def _audio_callback(self, indata, samples, time_received, status):
        # send audio data to server
        #print(time.strftime('%H:%M:%S'), "PI: stop recording: ", self._stop_recording)
        if not self._stop_recording:
            self._q.put(indata.copy())
            #print(time.strftime('%H:%M:%S'), "PI: recorded ", samples)
        
        return
    
    
    def _tcp_listener(self, s):
        # check if main loop should be stopped
        data = s.recv(4)
        if data.decode() == "STOP":
            self._stop_recording = True
            print(time.strftime('%H-%M-%S'), "PI: recording stopped.")
                #break   
        return
    
    def _find_mic_id(self, name):
        qd = sd.query_devices()
        i = 0
        while (i < len(qd)):
            if name in qd[i]['name']:
                return i
            i+=1
            
        return -1
    
    def _is_port_open(self, socket, ip, port):
        try:
            socket.connect((ip, port))
            return True
        
        except:
            return False
        
    def run(self):
        
        try:
            # get device index               
            #devices = sd.query_devices()
            mic_id = self._find_mic_id('USB')
            assert(mic_id > -1)
            
            # open TCP connection to server (laptop
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                print(time.strftime('%H-%M-%S'), "PI: Opened TCP client")
                while (not self._is_port_open(sock, self._ip, self._port)):
                    print(time.strftime('%H-%M-%S'), "PI: Waiting for TCP server to open...")
                    time.sleep(0.001)
                
                #sock.connect((self._ip, self._port))
                print(time.strftime('%H-%M-%S'), "PI: Connected to TCP server at address (", self._ip, ",", self._port, ")")
                
                # Wait for start sign
                data = sock.recv(5)
                if (data.decode() != "START"):
                    print(time.strftime('%H-%M-%S'), "PI: invalid message: ", data.decode())
                    return
    #                break
                self._start_time = time.time()
                
                # launch listener thread for STOP msg
                thr_tcp = threading.Thread(target=self._tcp_listener, args=[sock])
                thr_tcp.start()
                
                # open input stream and bind with callback fcn
                with sd.InputStream(samplerate=self._sample_rate,
                                    blocksize=self._block_size, device=int(mic_id),
                                    channels=1, dtype='float32',
                                    callback=self._audio_callback, clip_off=True):
                    
                    print(time.strftime('%H-%M-%S'), "PI: Opened audio stream")
                    # Loop until the server sends a message (to stop)
                    while (True):
                        if self._q.empty():
                            if (self._stop_recording):
                                print(time.strftime('%H-%M-%S'), "PI: all audio data sent.")
                                break  
                                
                        else:
                            #print(time.strftime('%H-%M-%S'), "PI: queue size = ", self._q.qsize())
                            sock.sendall(self._q.get())
                            self._q.task_done()
                        
                
                print(time.strftime('%H-%M-%S'), "PI: Audio stream closed")
                               
                self._q.join()
                thr_tcp.join()
                
                # tell server ssh can be closed
                #socket.sendall(b'STOP')
                
        except Exception as e:
            print(time.strftime('%H-%M-%S'), "PI: ", e)
            # write queue to file
            #filename = str(time.time()) + '.wav'
            #with sf.SoundFile(filename, mode='x', samplerate=self._sample_rate, channels=1) as file:
            #    while(not self._q.empty()):
            #        file.write(np.frombuffer(self._q.get(), dtype='float32'))
            #        self._q.task_done()
                    
            #    self._q.join()    
                
            #print("Written to file ", filename)
            #x, fs = sf.read(filename, dtype='float32')
            #print("Recorded ", len(x), " samples (t = ", len(x)/fs, ")")
            
        # close TCP
        print(time.strftime('%H-%M-%S'), "PI: Connection closed")
        
        return
    
# main program
parser = argparse.ArgumentParser(description='Set up the TCP connection to the server')
parser.add_argument('-i', '--ip', help='IP address of the TCP server', required=True)
parser.add_argument('-p', '--port', help='Port of the TCP server', required=True)
parser.add_argument('-s', '--samplerate', help='Microphone Sample Rate (Default: 48.0 kHz', required=False)
parser.add_argument('-b', '--blocksize', help='Number of samples received per callback (Default: 4096)', required=False) 
args = vars(parser.parse_args())

ip = args['ip']
port = int(args['port'])
fs = int(args['samplerate'])
buf = int(args['blocksize'])

client = TcpClientAudio(ip, port, fs, buf)

client.run()




