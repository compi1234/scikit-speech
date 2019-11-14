
""" libaudio
A small library for basic audio I/O based on PyAudio and some basic signal manipulation

Some inspriation was derived from a.o.:

https://www.swharden.com/wp/2016-07-19-realtime-audio-visualization-in-python/

https://github.com/AllenDowney/ThinkDSP/blob/master/code/thinkdsp.py

Notes:
    the buffers used by PyAudio are of type byte, but accomodating 
    different formats; taking care of this conversion is critical
    (.frombuffer() in record() and .tobytes() in play())
    
Conventions:
    we use databuffers or type np.float32 as standard interface with PyAudio
    we use a default data range of [-1.0 1.0] corresponding to full 16bit range of A/D
    normalization can be set on or off

History:
14/11/2018: single channel record(), play() 

"""

import pyaudio
import numpy as np

def record(fs=44100,time=1.0,Normalize=False):
    # record audio data with PyAudio
    # data is returned in np.int16 array

    RANGE16 = 32768.0
    CHUNK = 1024 # number of data points to read at a time
    # increase up to about 100 msec
    while CHUNK < fs/20:
        CHUNK = 2*CHUNK

    pa=pyaudio.PyAudio() # start the PyAudio class
    # create a numpy array holding a single read of audio data
    data = np.array([],np.int16)
    print('RECORDING for %.2f second(s)' % time)

    # open the input stream
    input=pa.open(format=pyaudio.paInt16,channels=1,rate=fs,input=True,
              frames_per_buffer=CHUNK) #uses default input device

    for i in range(int(time*fs/CHUNK)): 
        data = np.append(data,np.frombuffer(input.read(CHUNK),dtype=np.int16))
        print(".",end='')
              
    # close the stream gracefully
    input.stop_stream()
    input.close()
    pa.terminate()
    # convert to float in range [-1 1.0]
    data = data.astype(np.float32)/RANGE16
    if(Normalize):
        data = data/(1.01*max(abs(data)))
    return data


# play a signal (data) - single channel only
# the signal is normalized by default
def play(data,fs=44100,Normalize=True,amp=1.0):
    pa=pyaudio.PyAudio() # start the PyAudio class

    if( len(data)==0 ):
        print('Warning: play(libaudio) no data found')
        return
    
    if( type(data[0]) != np.float32 ):
        data = data.astype(np.float32)
     
    fmt= pyaudio.paFloat32
    ampx = max(abs(data))
    if((ampx > 1.0) & (not Normalize)):            
        print('Warning: play(libaudio) amp>1.0, forcing normalization') 
        Normalize = True
         
    if(Normalize):
        data = amp * data / (1.01*ampx)
            
    output = pa.open(format=fmt,
                channels=1,
                rate=fs,
                output=True)
    output.write(data.tobytes())
    output.stop_stream()
    output.close()
    pa.terminate()
    return

