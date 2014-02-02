import wave
import sys
import numpy
import struct
import math

if len(sys.argv) < 2:
    print("Extracts MFCCs from a wave file.\nUsage: %s filename.wav\n" % sys.argv[0])
    sys.exit(-1)

wf = wave.open(sys.argv[1], 'rb')

samp_rate = wf.getframerate()

if wf.getnchannels() != 1:
    print "Oops. This code does not work with multichannel recordings. Please provide a mono file\n"
    sys.exit(-1)

frames = wf.readframes(wf.getnframes())
data = numpy.array(struct.unpack_from("%dh" % wf.getnframes(), frames))
wf.close()

# NOTE: These frames are not the same 'frames' as the ones in the wav file 
frame_size = 0.025 #in seconds
frame_length = int(samp_rate*frame_size)
num_frames = int( math.ceil(data.shape[0]/float(frame_length)) )
data.resize(num_frames, frame_length)

# Let's apply a window to our frames 
hamm_win = numpy.hamming(frame_length)
data = data * hamm_win #Broadcasting should do the right thing

#Let's compute the spectrum
comp_spec = numpy.fft.rfft(data,n=1024)
mag_spec = abs(comp_spec)
