"""
analyse_noise_spectra.py
This script analyzes and plot the noise spectra for all the relevant measurement configurations

"""
#--------------------------------------------------
# Load modules
# -------------------------------------------------
import matplotlib.pyplot as plt
from . import functions as ft
from scipy.signal import welch
import numpy as np
import pickle

def get_noise_psd(dir, kid, pread, pw, filetype='med', filter=None, tqp=None, mph=None, mpp=None, nr_req_files=None, coord='circle', response='phase'):
    dir = dir.replace("\\", '/')
    if type(kid) is int:
        pulse_files, info_files = ft.get_files(dir + 'TD_Power', kid, pread, type=filetype)
    elif type(kid) is str:
        pulse_files = [dir + '/' + kid + '.bin']
        info_files = [dir + '/' + kid + '_info.dat']
    f0, Q, Qc, Qi, S21_min, dt, T =  ft.get_info(info_files[0])
    sff = 1 / dt / 1e6
    wl = round(pw * sff)
    nr_files = len(pulse_files)
    if nr_req_files is not None and nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = ft.get_data(pulse_files[:nr_req_files])
    signal = ft.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        sw = round(tqp * sff)
        window = ft.get_window(filter, sw)
        std = ft.get_sigma(signal, window)
        ph = mph*std
        pp = mpp*std
        noise_locs, _ = ft.find_pks(signal, ph, pp, window)
        signal_noises = ft.get_single_noises(signal, noise_locs, wl)
        fxx, nxx = welch(signal_noises, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
        print(nxx.shape)
        nxx = np.mean(nxx, axis=0)
    else:
        fxx, nxx = welch(signal, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
    return ft.logsmooth(fxx, nxx)


def noise_spectra(path2data, name, pw=2000000, filetype='med', filter=None, exclude_dc=True):

    #--------------------------------------------------
    # Load kid_dict
    # -------------------------------------------------
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)
    print("Loaded file %s" % path2kid_dict)

    #--------------------------------------------------
    # Analysis
    # -------------------------------------------------
    keys = ['BF dark', '3.8um off', '3.8um', '8.5um', '18.5um off', '18.5um', 'ADR dark', '25um off', '25um']
    for key in keys:
        kid = kid_dict[name][key]['kid']
        pread = kid_dict[name][key]['pread']
        path = kid_dict[name][key]['dir']
        freqs, nxx = get_noise_psd(path, kid, pread, pw, filetype=filetype, filter=filter)
        kid_dict[name][key]['noise'] = [freqs, nxx]

    with open(path2kid_dict, 'wb') as f:
        pickle.dump(kid_dict, f)
    print("Updated file %s" % path2kid_dict)

    #--------------------------------------------------
    # Plotting
    # -------------------------------------------------
    keys = ['BF dark', None, None, '3.8um off', '3.8um', None, '8.5um', '18.5um off', '18.5um', 'ADR dark', '25um off', '25um']
    labels = ['dark DR', ' ', ' ', '3.8 µm; $T_\mathrm{lab}=293\ K$', '3.8 µm; $T_\mathrm{lab}=293\ K$, QTH', ' ', '8.5 µm; $T_\mathrm{lab}=293\ K$', 
            '18.5 µm; $T_{bb}=3\ K$', '18.5 µm; $T_{bb}=160\ K$', 'dark ADR', '25 $\mu$m, $T_{bb}=3\ K$', '25 $\mu$m, $T_{bb}=24\ K$']
    colors = ['k','None','None', 'b', 'b','None', 'y', 'o', 'o', 'k', 'p', 'p']
    linestyles = ['-', ' ', ' ', '-', '--', ' ', '--', '-', '--', '-', '-', '--']

    fig, axes = plt.subplot_mosaic('a;b', figsize=(18.5/1.85/2.54, 9/2.54), sharex=False, sharey=True, constrained_layout=True)
    ylabel_psd = '$S_{\\theta}/(4Q_l)^2$ [dBc/Hz]'
    xlabel_psd = 'Frequency [Hz]'
    ylim_psd = [-175, -145]
    xlim_psd = [10, 2e4]
    xticks = np.logspace(1, 4, 4, endpoint=True)
    width = 1

    for i, key in enumerate(keys):
        if i < len(labels)-3:
            ax = axes['a']
        else:
            ax = axes['b']
            ax.set_xlabel(xlabel_psd)
        if key:
            [f, s] = kid_dict[name][key]['noise']
            Q = kid_dict[name][key]['Q']
            ax.semilogx(f[exclude_dc:],  10*np.log10(s[exclude_dc:]/(4*Q**2)), label=labels[i], c=colors[i], ls=linestyles[i])
        else:
            ax.semilogx([],  [], label=labels[i], c=colors[i], ls=linestyles[i])
        ax.set_xlim(xlim_psd)
        ax.set_ylim(ylim_psd)
        ax.set_ylabel(ylabel_psd)
        ax.set_xticks(xticks, minor=False)
        ax.legend(loc='upper right', ncols=3, borderaxespad=0., frameon=False, handlelength=1, columnspacing=0.25)

    plt.savefig('figures/%s_noise_spectra.pdf' % name)
    plt.show()

if __name__ == "__main__":
    noise_spectra()