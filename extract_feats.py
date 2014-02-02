import wave
import sys
import numpy
import struct
import math

if len(sys.argv) < 2:
    print("Extracts MFCCs from a wave file.\nUsage: %s filename.wav\n" % sys.argv[0])
    sys.exit(-1)


def gen_mel_filts(num_filts, framelength, samp_freq):
    mel_filts = numpy.zeros((framelength, num_filts))
    step_size = int(framelength/float((num_filts + 1))) #Sketch it out to understand
    filt_width = math.floor(step_size*2)
    
    filt = numpy.bartlett(filt_width)
    
    step = 0
    for i in xrange(num_filts):
        mel_filts[step:step+filt_width, i] = filt
        step = step + step_size

    # Let's find the linear filters that correspond to the mel filters
    # The freq axis goes from 0 to samp_freq/2, so...
    samp_freq = samp_freq/2 

    filts = numpy.zeros((framelength, num_filts))
    for i in xrange(num_filts):
        for j in xrange(framelength):
            freq = (j/float(framelength)) * samp_freq

            # See which freq pt corresponds on the mel axis
            mel_freq = 1127 * numpy.log( 1 + freq/700  )
            mel_samp_freq = 1127 * numpy.log( 1 + samp_freq/700  )

            # where does that index in the discrete frequency axis
            mel_freq_index = int((mel_freq/mel_samp_freq) * framelength)
            if mel_freq_index >= framelength-1:
                mel_freq_index = framelength-1
            filts[j,i] = mel_filts[mel_freq_index,i]

    # Let's normalize each filter based on its width
    for i in xrange(num_filts):
        nonzero_els = numpy.nonzero(filts[:,i])
        width = len(nonzero_els[0])
        filts[:,i] = filts[:,i]*(10.0/width)

    return filts




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


# Mel warping
filts = gen_mel_filts(25, 513, samp_rate) # 1024 point FFT
mel_spec = numpy.dot(mag_spec,filts)

# Mel log spectrum
mel_log_spec = mel_spec #trust me on this
nonzero = mel_log_spec > 0
mel_log_spec[nonzero] = numpy.log(mel_log_spec[nonzero])

