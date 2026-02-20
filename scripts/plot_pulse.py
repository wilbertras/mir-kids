"""
analyse_time_domain.py

This scripts plots the time domain response of the detector for the differenc measurement configurations
"""

#--------------------------------------------------
# Load modules
# -------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch
import pickle

from . import functions as ft

#--------------------------------------------------
# Define functions
# -------------------------------------------------
def get_filtered_response(dir, kid, pread, nr_req_files, filter, lifetime, coord='circle', response='phase', temp='', start=0):
    dir = dir.replace("\\", '/') + 'TD_Power/'
    if type(kid) is int:
        pulse_files, _ = ft.get_files(dir, kid, pread, type='vis')
    elif type(kid) is str:
        pulse_files = [dir + '/' + kid + '.bin']
    nr_files = len(pulse_files)

    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = ft.get_data(pulse_files[start:start+nr_req_files])
    signal = ft.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = ft.get_window(filter, lifetime)
        filtered_signal = fftconvolve(signal, window, mode='valid')
        return signal, filtered_signal
    else:
        return signal, signal

def noise_psd(signal):
    return welch(signal, fs=int(1e6), window='hamming', nperseg=1000)

def photon_timestreams(path2data, name, lifetime=50, filter='exp', nr_req_files=3, start=10):
    #--------------------------------------------------
    # Load kid_dict
    # -------------------------------------------------
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)
    print("Loaded file %s" % path2kid_dict)

    #--------------------------------------------------
    # Analyse time domain
    # -------------------------------------------------
    keys = ['BF dark', '3.8um off', '3.8um', '8.5um', '18.5um off', '18.5um', '25um off', '25um']
    for i, key in enumerate(keys):
        kid = kid_dict[name][key]['kid']
        pread = kid_dict[name][key]['pread']
        path = kid_dict[name][key]['dir']
        Q_bf_dark = kid_dict[name][key]['Q']
        td, filtered_td = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime, start=start)
        kid_dict[name][key]['timestream'] = [td, filtered_td]

    #--------------------------------------------------
    # Plot time domain
    # -------------------------------------------------
    keys = [['BF dark'], ['3.8um', '3.8um off'], ['8.5um'], ['18.5um', '18.5um off'], ['25um', '25um off']]
    titles = ['Dark', '$3.8$ $\mu m$', '$8.5$ $\mu m$', '$18.5$ $\mu m$', '$25$ $\mu m$']
    labels = [['dark DR'], ['$T_\mathrm{lab}=293\ K$, QTH', '$T_\mathrm{lab}=293\ K$'], ['$T_\mathrm{lab}=293\ K$'], ['$T_{bb}=160\ K$', '$T_{bb}=3\ K$'], ['$T_{bb}=24\ K$', '$T_{bb}=3\ K$']]
    Q_bf_dark = kid_dict[name]['BF dark']['Q']
    colors = 'kbyop'
    offset = 0.5

    fig, axes = plt.subplots(1, 5, figsize=(18.5/2.54,6/2.54), sharex=True, sharey=True, constrained_layout=True)
    ylabel_response = 'Phase response [rad]'
    xlabel_response = 'Time [s]'
    ylim = [-.75, 1.5]
    width = .05
    step = .25
    yticks = np.arange(ylim[0], ylim[1]+step, step)
    filtered = True

    for i, key in enumerate(keys):
        ax = axes[i]
        wl = key[0]
        scale = Q_bf_dark/kid_dict[name][wl]['Q']
        td = kid_dict[name][wl]['timestream'][filtered] * scale
        t = np.linspace(0, nr_req_files, len(td))
        ax.plot(t, td, lw=width, c=colors[i])
        ax.plot([], [], lw=1, c=colors[i], label=labels[i][0])
        if len(key)==2:
            wl = key[1]
            scale = Q_bf_dark/kid_dict[name][wl]['Q']
            td = kid_dict[name][wl]['timestream'][filtered] * scale
            ax.plot(t, td-offset, lw=width, c='gray')
            ax.plot([], [], lw=1, c='gray', label=labels[i][1])
        ax.set_xlabel(xlabel_response)
        ax.set_xlim([0, nr_req_files])
        ax.set_yticks(yticks)
        ax.set_ylim(ylim)
        ncols = 1
        ax.legend(loc='upper center', ncols=ncols, mode="expand", borderaxespad=0., frameon=False, handlelength=1.5)
        ax.set_title(titles[i])
    axes[0].set_ylabel(ylabel_response)

    plt.savefig('figures/%s_time_domain.pdf' % name)
    plt.show()

if __name__ == "__main__":
    photon_timestreams()