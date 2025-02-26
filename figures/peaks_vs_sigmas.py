import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, fftconvolve
import scipy.constants as sc
from scipy.optimize import curve_fit
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figures.functions as f
from main import pulse_analysis
import figures.matplotlibcolors as matplotlibcolors
from filters.filters import load_all_filters

def count_pulses(dir, kid, pread, nr_req_files, filter, lifetime, mph, mpp, coord='circle', response='phase'):
    dir = dir.replace("\\", '/')
    pulse_files, _ = f.get_files(dir, kid, pread, type='vis')
    nr_files = len(pulse_files)

    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = f.get_data(pulse_files[:nr_req_files])
    signal = f.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = f.get_window(filter, lifetime)
    std = f.get_sigma(signal, window)
    ph = mph*std
    pp = mpp*std
    if len(window):    
        signal = fftconvolve(signal, window, mode='valid')
        window_offset = int(np.argmax(window[::-1]))
    else:
        signal = signal
        window_offset = 0
    # locs, _ = f.find_pks(signal, ph[0], pp, window)
    # total_pulses = len(locs)
    fig, ax = plt.subplots()
    ax.hist(signal, bins='auto')
    ax.axvline(ph[0])
    ax.axvline(-ph[0])
    plt.show()
    # return total_pulses, nr_req_files

kid = 5
pread = 113
file_type = 'vis'
pw = 1500
pw_offset = 100
filter = 'exp'
lifetime = 250
tqp = [200, 600]
iterate = 0


dir = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800 long\TD_Power"
dir = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off\TD_Power"
chuncksize = 10
nr_chuncks = 1
mph = np.array([5, 40])
mpp = mph[0]

nr_pulses, nr_analyzed_files = count_pulses(dir, kid, pread, chuncksize*nr_chuncks, filter, lifetime, mph, mpp, coord='circle', response='phase')