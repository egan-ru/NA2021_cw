#!/usr/bin/python
#thermal physics lab1
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import os.path as path
import scipy.signal as signal
import math
import time

#used to print debug extras
NA_DEBUG = 14

#carrier freq
CARRIER_FREQ = 10000000000

#set reference signal to nearest neighbor, if zero use freq-based
NN_ENABLE = 0

#file with samples
FILENAME = "DSP_02_FD30M.txt"

#sample freq
SAMPLE_FREQ = (30 * 1000000)

#intermediate frequency
IMED_FREQ = (SAMPLE_FREQ/4)

#signal period
PERIOD = (180 / 1000000)

#duty cycle
DUTY = 5

#pn-polynom
PN_POLYNOM = [5, 3, 0]
PN_ORDER = 5
PN_LENGTH = (2**5) - 1
PN_STATE = [1, 0, 0, 0, 0]

#length of one sample
TAU = (PERIOD / (DUTY * PN_LENGTH))

#nearest-neighbor constants

#freq multiplier for lowered freq
LOW_FREQ_MULT = (SAMPLE_FREQ / IMED_FREQ)
LOW_FREQ_DIV = (IMED_FREQ/SAMPLE_FREQ)

#found threashold in coherent accumulation
CH_THRESHOLD = 3


#wave speed  in m/s
WAVE_SPEED = 300000000


"""
used to close timer window after some time
"""
def plot_close():
    plt.close()

"""
plot debug abs of fft data,
used to control reading the data - NA lookup
@param data     - data to plot
"""
def plot_fft_data(data, time):
    fig = plt.figure()
    tmr = fig.canvas.new_timer(interval = time*1000, callbacks = [(plot_close, [], {})])
    fft = np.fft.fft(data)
    afft = np.abs(fft)
    plt.xlabel("freq, Hz")
    plt.ylabel("amp")
    plt.plot(afft, 'b')
    plt.show()

"""
calculate sampledata parameters
@param scale        - scale factor (should be 4 ... 6)

@return             - samplecount and sample freq
"""
def signaldata_get(scale):
    global SAMPLE_FREQ
    global PERIOD
    global PN_LENGTH
    global TAU

    if(4 > scale or 6 < scale):
        print("low freq mult is ", scale,"this is out of range 4 .. 6, low freq mult will be set to 4")
        scale = 4

    #period in samples
    #sample_time = 1/SAMPLE_FREQ
    #period_s = round(PERIOD/sample_time)
    #print("period contain ", period_s, " samples") 

    #set lowered discretization time and freq
    samp_lo_time = TAU/scale
    #count of samples with lowered freq
    samplecount = round(PERIOD/samp_lo_time)&(~0x03)
    #сlarify values after samplecount calculation
    samp_lo_freq = samplecount/PERIOD
    samp_lo_time = 1/samp_lo_freq

    print("lowered period : ", '%.f4' %(samp_lo_time*1000000), " usec")
    print("lowered freq :   ", '%.f4' %(samp_lo_freq/1000000), " MHz")

    return (samplecount, samp_lo_freq)


"""
solve na_2021 by nearest neighbor method
@param pnseq        - pn sequence
@param samplecount  - estimated number of samples

@return             - signal and spectre of refernece signal
"""
def nearest_neighbor(pnseq, samplecount):
    global SAMPLE_FREQ
    global PERIOD
    global PN_LENGTH
    global TAU

    samp_lo_freq = samplecount/PERIOD

    #calc last boundary in sample
    lastnum = math.floor(PN_LENGTH*samp_lo_freq*TAU)

    nn_signal_s = np.zeros(samplecount)

    index_translator = 1/(samp_lo_freq*TAU)
    for i in range(lastnum):
        sel = math.floor(i*index_translator)
        nn_signal_s[i] = pnseq[sel]
        
    if(2 == NA_DEBUG):
        plt.xlabel("sample")
        plt.ylabel("level")       
        plt.plot(nn_signal_s)
        plt.show()

    nn_spectre = np.fft.fft(nn_signal_s)    
    return (nn_signal_s.reshape(samplecount), nn_spectre.reshape(samplecount))


"""
solve na_2021 by nearest neighbor method
@param pnseq        - pn sequence
@param samplecount  - estimated number of samples

@return             - signal and spectre of reference signal
"""
def freq_based(pnseq, samplecount):
    global SAMPLE_FREQ
    global PERIOD
    global PN_LENGTH
    global TAU

    fb_signal_s = np.empty(samplecount) 
    mid = samplecount>>1
    
    fb_signal_s[:mid] = np.arange(0,mid)
    fb_signal_s[mid:] = np.arange(-mid,0)
    #optimize fb_signal = (fb_signal * samp_lo_freq)/samplecount,
    #where samp_lo_freq is samplecount/period
    fb_signal_s /= PERIOD

    if(3 == NA_DEBUG):
        plt.xlabel("sample")
        plt.ylabel("level")       
        plt.plot(fb_signal_s)
        plt.show()

    #reshape signal to simlify arith
    fb_signal_s = fb_signal_s.reshape(1,samplecount)
    #reshape pnseq to simplify arith
    pnseq = pnseq.reshape(1,PN_LENGTH)
    #pulses moved on 0.5 to make first pulse front on start of coordinates
    timeline = (np.arange(0, PN_LENGTH) + 0.5)*TAU
    #reshape timeline
    timeline = timeline.reshape(PN_LENGTH,1)

    timefreq = np.dot(timeline, fb_signal_s)
    delay_part = np.exp(-2j*np.pi*timefreq)
    delay_part = np.dot(pnseq, delay_part)

    signal_part = np.sinc(fb_signal_s*TAU)

    if(4 == NA_DEBUG):
        print("shape of signal part : ", signal_part.shape)
        print("shape of delay part  : ", delay_part.shape)

    fb_spectre = signal_part*delay_part

    fb_signal = np.fft.ifft(fb_spectre)
    if(4 == NA_DEBUG):
        plt.xlabel("sample")
        plt.ylabel("level")       
        plt.plot(np.real(fb_signal.reshape(samplecount,1)))
        plt.show()
    
    return (fb_signal.reshape(samplecount), fb_spectre.reshape(samplecount))



"""
calculate packet spectre
used in freq method
@param tau      - length of one sample
@param pn_list  - list to encode
@param freq     - selected freq

@return         - packet spectre
"""
def packet_spectre(tau, pn_list, freq):
    mult = tau * np.sinc(np.pi*tau*freq)
    result = 0j
    pn_length = pn_list.length
    for i in range(pn_length):
        result += pn_list[i]*exp(-1j*2*np.pi*(i + 0.5)*tau*freq)

    result = result * mult
    return result

"""
get sampledata from NA file
@param filename     - file to open

@return             - sampledata if file opened, None else
                      
"""
def filedata_get(filename):
     #check that file present and check that it's size is multiple of 4
    print("Opening file " + filename)
    if not path.isfile(filename):
        print (filename + " not exist")
        return None

    size = os.stat(filename).st_size
    entries = size >> 2
    print("file size = ", '%d' %size, ' B', "\ncontain ", '%d' %entries, " 4B single entries")

    if(0x03 & size):
        print("bad size: should be mutiple of 4")
        return None

    #load binary into nympy array
    try:
        na_data = np.fromfile(filename, dtype=np.single)
    except:
        print("undefined IO error")
        return None

    return na_data


"""
move count/2 samples from pos (center - count/2 : center) of period to left corner
move count/2 samples from pos (center: center + count/2) iod to right corner
@param period   - sample data to proceed
@param center   - center pos to take samples
@param count    - count of samples to move

@return         - moved spectre
"""
def move_spectre(sampledata, center, count):
    size = sampledata.size 

    offset = count >> 1
    in_spectre = np.fft.fft(sampledata)
    out_spectre = np.concatenate((in_spectre[center:center + offset], in_spectre[center - offset: center]))

    if(6 == NA_DEBUG):
        fig, plots = plt.subplots(2)
        fig.suptitle("input and moved spectre")

        plots[0].plot(np.fft.fft(sampledata))
        plots[1].plot(out_spectre)
        plt.show()

    return out_spectre

"""
make pulse compression
@param spectre_ref      - reference sigal spectre
@param spectre_signal   - signal spectre 

@return                 - impulse
"""
def pulse_compression(spectre_ref, spectre_signal):
    cv_ref = np.conj(spectre_ref)
    #print("size of spectre_ref: ", spectre_ref.size)
    #print("size of spectre_signal: ", spectre_signal.size)

    cp_spectre = cv_ref*spectre_signal
    cp_pulse = np.real(np.fft.ifft(cp_spectre))

    if(7 == NA_DEBUG):
        plt.suptitle("compressed impulse")
        plt.xlabel("sample")
        plt.ylabel("level")       
        plt.plot(abs(cp_pulse))
        plt.show()      

    return cp_pulse


"""
make coherent accumulation of input data,
@param na_data          - input samples, sliced by period into matrix
@param pos              - pos for spectre moving(in every line)
@param ref_spectre      - spectre of reference signal

@return                 - accumulated data
"""
def coherent_accumulation(na_data, pos, ref_spectre):
    #process in slow-time the input data
    na_pr_count = na_data.shape[0]
    samplecount = ref_spectre.shape[0]

    na_proc_array = np.empty((na_pr_count, samplecount))
    for i in range(na_pr_count):
        na_data_period = na_data[i]
        line_spectre = move_spectre(na_data_period, pos, samplecount)
        pulse = pulse_compression(ref_spectre, line_spectre)
        na_proc_array[i] = pulse

    #reshape matrix and make fft in every line, then reshape back (for coherent burst accumulation)
    coh_acc_na_array = na_proc_array.T
    for i in range(samplecount):
        coh_acc_na_array[i] = np.fft.fft(coh_acc_na_array[i])

    coh_acc_na_array = coh_acc_na_array.T
    coh_acc_na_array = np.abs(coh_acc_na_array)

    return coh_acc_na_array


"""
check, that signal present in data
@param coh_acc_data     - coherent accumulated data
@param threshold        - signal thrshold

@return                 - signal coordinates, if present or null, if not
"""
def signal_is_present(coh_acc_data, threshold, dbg_num):
 
    #number of samples, that more then threashold should be 1
    lmax = np.amax(coh_acc_data) 

    if(threshold > lmax):
        #we have not peak
        return None

    #we should not have lot of peaks with half of lmax amplitude
    threshold_mid = lmax/4
    cmax = (coh_acc_data > threshold_mid).sum()
    #if(4 < cmax):
    #    #we have too wide spectrum
    #    return None

    #we have peak - return it's value and index
    if(11 == NA_DEBUG):
        maxvalues = np.amax(coh_acc_data, axis=0)
        suptitle = "offset num : " + str(debug_num) + "\npulses more then threshold : " + str(cmax) + "\nmaxvalue : " + str(lmax)
        plt.suptitle(suptitle)
        plt.xlabel("sample")
        plt.ylabel("level")       
        plt.plot(abs(maxvalues))
        plt.show()      

    if(12 == NA_DEBUG):
        plot_array = coh_acc_data
        #print surface plot for debugging
        suptitle = "offset num : " + str(dbg_num) + "\npulses more then threshold : " + str(cmax) + "\nmaxvalue : " + str(lmax)
        fig = plt.figure()
        plt.suptitle(suptitle)
        surf3d = fig.add_subplot(111, projection='3d')
        X = np.arange(0, na_pr_count, 1)
        Y = np.arange(0, samplecount, 1)
        X, Y = np.meshgrid(X,Y)
        Z = plot_array[X,Y]  

        #print("shape of Z is ", Z.shape)

        surf3d.plot_surface(X, Y, Z, cmap=plt.cm.get_cmap('hsv'))
        plt.show()

    na_argmax = np.unravel_index(np.argmax(coh_acc_data, axis=None), coh_acc_data.shape)
    return (lmax, na_argmax[1], na_argmax[0])   


# Nikolay Alexandrowich 2021 task
if __name__ == '__main__':
    if not (sys.version_info.major == 3 and sys.version_info.minor >= 7):
        print("This script requires Python 3.7 or higher!")
        print("You are using Python {}.{}.".format(sys.version_info.major, sys.version_info.minor))
        sys.exit(1)

    print("NA begin\n")

    na_data = filedata_get(FILENAME)
    if(na_data is None):
        print ("NA fail")
        sys.exit(os.EX_SOFTWARE)
   
    if(1 == NA_DEBUG):
        print("\nSample count:", na_data.size)
        plot_fft_data(na_data, 2)

    print("generating pn-sequence, params:\n\torder", PN_ORDER,"\n\tpolynom", PN_POLYNOM, "\n\tstart state", PN_STATE,"\n\tlength", PN_LENGTH, "\n\t")
    pnseq = signal.max_len_seq(nbits=PN_ORDER, state=PN_STATE, length=PN_LENGTH, taps=PN_POLYNOM)[0]
    print("pn sequence: ", pnseq)
    #tranlate OOK sequence into FM sequence (1 into 1, 0 into -1)
    pnseq = (pnseq << 1) - 1

    print("pn sequence: ", pnseq)

    #length of one sample
    print("calculated tau, ms: ", '%.4f' %(TAU * 1000000))

    #period in samples
    sample_time = 1/SAMPLE_FREQ
    period_s = round(PERIOD/sample_time)
    print("period contain ", period_s, " samples") 

    signaldata = signaldata_get(LOW_FREQ_MULT)
    samplecount = signaldata[0]
    sample_lo_freq = signaldata[1]

    if(1 == NN_ENABLE):
        # method1: nearest neighbor
        print("\nmethod1: nearest neighbor\n")
        refsignal = nearest_neighbor(pnseq, samplecount)
    else:
        print("\nmethod2: freq-based\n")
        refsignal = freq_based(pnseq, samplecount)

    ref_signal = refsignal[0]
    ref_spectre = refsignal[1]

    #slice data by samplecount chunks for further processing
    na_data_len = na_data.size
    na_pr_count = round(na_data_len/period_s)

    #check if we can't make roudly from data matrix (period_s, length/period_s)
    if(na_pr_count*period_s != na_data_len):
        #data have tail - cut off tail
        samples_dropped = na_data_len - na_pr_count*period_s 
        print("data have tail. ", samples_dropped, " samples was dropped")
        na_data = na_data[:na_pr_count*period_s]
    else:
        #data have not tail
        print("data have not tail")

    #length of all input signal, used when we will calculate the position
    total_time = (na_pr_count*period_s)/SAMPLE_FREQ

    #reshape input data
    na_data = na_data.reshape(na_pr_count,period_s)
    print("reshaped na_data have shape ", na_data.shape) 

    if(5 == NA_DEBUG):
        #one period of data from start of NA file (transmitted data)
        na_data_period = na_data[0]
        #moved input spectre
        mi_spectre = move_spectre(na_data_period, round(LOW_FREQ_DIV*period_s), samplecount)
        #pulse compression
        pulse = pulse_compression(ref_spectre, mi_spectre)
        #print first period for debugging 
        plt.xlabel("sample")
        plt.ylabel("level")       
        plt.plot(np.fft.fft(na_data_period))
        plt.show()      


    #repeatly procedd coherent accumulation for shifted spectre
    pos = round(LOW_FREQ_DIV*period_s)

    #get point of input signal
    starttime = time.time()
    fp_data = coherent_accumulation(na_data, pos, ref_spectre)
    founded = signal_is_present(fp_data, CH_THRESHOLD, 0)
    end_time = round((time.time() - starttime)*1000000)

    if founded is None:
        print("tx peak is not founded, signal was not transmitted")
        sys.exit(os.EX_SOFTWARE)

    fp_peak_max = founded[0]
    fp_peak_pos = founded[1]
    print("first peak max is ", '%.3f' %fp_peak_max, " located at ", founded[1], " sample\none freq processing take ", end_time, " useconds")


    #we shifte the spectre right and left by samplecount/8
    shift_max = samplecount >> 3

    starttime = time.time()   
    #used to find the value with max peak
    max_pos = 0
    max_value = 0
    max_freqshift = 0
    max_freq = 0

    #step one - spectre is shifted to right
    for freqshift in range (-shift_max, +shift_max):
        moved_ref_spectre = np.roll(ref_spectre, freqshift) 

        cacc_data = coherent_accumulation(na_data, pos, moved_ref_spectre)

        #cutoff transmitted signal (it's send at zero time and it's petals 
        cacc_data[range(10)] = 0

        founded = signal_is_present(cacc_data, CH_THRESHOLD, freqshift)
        if founded is not None :
            peak_max = founded[0]
            peak_pos = founded[1]
            peak_freq = founded[2]
            if(13 == NA_DEBUG):
                #don't print peak info, if we meashuring the time of calculations
                print("peak found.\n\tvalue : ", '%.4f' %peak_max, "\n\tpos : ", peak_pos, "\n\tfreqshift : ", freqshift)
            if(max_value < peak_max):
                max_value = peak_max
                max_pos = peak_pos
                max_freqshift = freqshift
                max_freq = peak_freq


    time_passed = round((time.time() - starttime)*1000000)

    #ok, we get the data
    if(0 < max_value):
        #our peaks is located reapetedly in timeline, so if tx peak located is after the rx peak,
        # than we need to add the period to delta (delta should not be negative)
        sample_delta = max_pos - fp_peak_pos
        if(0 > sample_delta):
            sample_delta += samplecount

        lowered_discretization_freq = (samplecount/period_s)*SAMPLE_FREQ
        time_delta = sample_delta/lowered_discretization_freq

        #signal move to target and back, so only first part of time should be takes, when we calculate the range
        target_range = WAVE_SPEED*time_delta/2

        freq_base = CARRIER_FREQ
        print("freq_base : ", freq_base)
        freq_doppler = (peak_freq/period_s + max_freqshift)/PERIOD
        print("freq_doppler : ", freq_doppler)
        #freq_doppler = (20 + 120/128)/PERIOD

        #calculate speed from doppler effect, using radar formula = dF = 2VF0/c ->
        target_speed = (WAVE_SPEED * freq_doppler) / (2 * freq_base) 

        print("located signal present\n\tat_range: ", round(target_range))
        if((0.01) < target_speed):
            print("\n\tapproaching with speed: ", '%.3f'%target_speed)
        elif((-0.01) > target_speed):
            print("\n\tretreating with speed: ", '%.3f'%abs(target_speed))
        else:
            print("\n\tno target motion")
    else:
        print("target not present")


    print("calculation take :", time_passed, "useconds")

    print("NA success")
    sys.exit(os.EX_OK)

