"""
analyse_efficiency.py

This module contains functions for analyzing the efficiency based on pulse data. 
"""


#--------------------------------------------------
# Import modules
# -------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, medfilt
import scipy.constants as sc
from scipy.optimize import curve_fit
import pickle
import functions as ft
import matplotlibcolors
plt.style.use('matplotlibrc')


#--------------------------------------------------
# Load kid_dict
# -------------------------------------------------
with open('path2data.txt', 'r') as file:
    path2data = file.readlines()[0]
path2kid_dict = r'%skid_dict.pkl' % path2data
with open(path2kid_dict, 'rb') as f:
    kid_dict = pickle.load(f)
print("Loaded file %s" % path2kid_dict)


#--------------------------------------------------
# Define functions
# -------------------------------------------------

def count_pulses(dir, kid, pread, lifetime, mph, mpp, filter='exp', nr_req_files=None, coord='circle', response='phase', file_type='vis'):
    """
    This function simply counts the number of pulses. It does not distinguish between single pulses and pileup events
    """
    dir = dir.replace("\\", '/')
    pulse_files, info_files = ft.get_files(dir, kid, pread, type=file_type)
    f0, Q, Qc, Qi, S21_min, dt, T = ft.get_info(info_files[0])
    sff = round(1 / dt) / 1e6
    nr_files = len(pulse_files)
    if nr_req_files is not None and nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = ft.get_data(pulse_files[:nr_req_files])
    signal = ft.coord_transformation(phase, amp, coord=coord, response=response)
    medsignal = medfilt(signal)
    time = len(signal) * dt
    if filter:
        window = ft.get_window(filter, lifetime)
    locs, _ = ft.find_pks(medsignal, mph[0], mpp, window)
    total_pulses = len(locs)
    return total_pulses, time    


def get_bp(theta, mph):
    """
    This function determines the bandwidth of the filter stack by finding the peaks in the transmission spectrum and determining their widths at a certain threshold
    """
    locs, _ = find_peaks(theta, height=mph, prominence=mph)
    _, _, lefts, rights = peak_widths(theta, locs, rel_height=.999)
    fig, ax = plt.subplots()
    ax.loglog(wls*1e6, theta, label='total transmission')
    ax.axhline(mph, ls='--', c='k', label='threshold')  
    ax.set_xlim([1, 1e2])
    ax.set_xlabel('Wavelength [µm]')
    ax.set_ylabel('Transmission')
    ax.legend()
    if len(locs) == 1:
        bp = np.arange(np.floor(lefts[0]), np.ceil(rights[0]), dtype=int)
        ax.plot(wls[bp]*1e6, theta[bp], c='r', label='bandpass')    
        plt.plot()
        return bp
    else:
        plt.plot()
        raise ValueError('More than one peak found')


def counts_vs_temp(dir, temps, kid, pread, lifetime, mph, mpp):
    """
    This function counts the number of pulses for each temperature and returns a dictionary with the temperatures and the corresponding photon count rates. 
    """
    dict = {}
    photonrates = []
    for temp in temps:
        path = dir.replace("160", str(temp)) + 'TD_Power/'
        nr_pulses, time = count_pulses(path, kid, pread, lifetime, mph, mpp, coord='circle', response='phase')
        Nph = nr_pulses/time
        photonrates.append(Nph)
        print('Temp: %d K, Nph: %.1f' % (temp, Nph))
    dict['Tbb'] = np.array(temps)
    dict['Nph'] = np.array(photonrates)
    return dict


def photon_rate(temps, eta):
    """
    This is the fitting function for the efficiency eta. It computes the total radiated photon rate from the total tranmission spectrum and the Planck spectrum.    
    """
    wavelength = float(wl[:-2])*1e-6                   
    Eph = sc.h*sc.c/wavelength                                  
    planck = ft.planck_wl(wavelength, np.array(temps).reshape((-1,1)))
    radiance = theta*planck
    power = np.trapezoid(radiance[:,bandpass], wls[bandpass])
    photon_rate = power / Eph
    return eta*photon_rate


def fit_efficiency(dict, fit_idx):
    """
    This function find the efficiency eta from fitting the expected photonrates to the detected photonrates as a function of temperature. Select the fitting range with fit_idx
    """
    temps = dict['Tbb']
    Nph = dict['Nph']
    [eta], pcov = curve_fit(photon_rate, temps[fit_idx[0]:fit_idx[1]], Nph[fit_idx[0]:fit_idx[1]])
    uncertainty = np.sqrt(np.diag(pcov))[0]
    dict['eta'] = [eta, uncertainty]
    print('Fit efficiency: %.2f +- %.2f' % (eta*100, uncertainty*100))
    dict['fit index'] = fit_idx   
    return dict

#--------------------------------------------------
# Initiate parameters
# -------------------------------------------------
name = 'KID26'
wl = '18.5um'                           
dir = kid_dict[name][wl]['dir']
kid = kid_dict[name][wl]['kid']          
pread = kid_dict[name][wl]['pread']
std = kid_dict[name][wl]['std']
nr_std = kid_dict[name][wl]['stds'][-1]
lifetime = kid_dict[name][wl]['tqp']                                      
redo_pulse_detection = False                                     # set to True to redo the pulse count analysis, if False only the fitting is redone

mph = np.array([nr_std*std, 3])                         # pulse detection threshold taken exactly equal to the one used to determine the resolving power
mpp = mph[0]                                            # minimal peak prominence, set equal to minimal peak height
temps = [3, 30, 50, 85, 100, 135, 142, 150, 160, 180]   # range of rediator temperatures to analyse
[wls, theta] = kid_dict[name][wl]['theta']              # total transmission spectrum of the filter stack at the wavelength of interest                           
bandpass = get_bp(theta, 1e-19)                         # indices indicating the bandpass of the total filterstack

#--------------------------------------------------
# Photon count analysis
# -------------------------------------------------
if 'Nph_vs_Tbb' not in kid_dict[name][wl] or redo_pulse_detection:
    kid_dict[name][wl]['Nph_vs_Tbb'] = counts_vs_temp(dir, temps, kid, pread, lifetime, mph, mpp)
    with open(path2kid_dict, 'wb') as f:
        pickle.dump(kid_dict, f)
    print("Updated file %s" % path2kid_dict)

#--------------------------------------------------
# Fit efficiency
# -------------------------------------------------
dict = fit_efficiency(kid_dict[name][wl]['Nph_vs_Tbb'], fit_idx=[3,None])
kid_dict[name][wl]['Nph_vs_Tbb'].update(dict)

#--------------------------------------------------
# Save kid_dict
# -------------------------------------------------
with open(path2kid_dict, 'wb') as f:
    pickle.dump(kid_dict, f)
print("Updated file %s" % path2kid_dict)

#--------------------------------------------------
# Plot efficiency
# -------------------------------------------------
temps = kid_dict[name][wl]['Nph_vs_Tbb']['Tbb']
Nph = kid_dict[name][wl]['Nph_vs_Tbb']['Nph']
eta = kid_dict[name][wl]['Nph_vs_Tbb']['eta'][0]    
Nph_dark = kid_dict[name][wl]['dcr']['4']*1e-3

fig, ax = plt.subplots(figsize=(18.5/2/2.54, 7/2.54), constrained_layout=True)
xlim = [0, 200]
ylim = [1e-2, 1e2]
yerr = np.sqrt(Nph)
ax.errorbar(temps, Nph, yerr=yerr, linestyle='None', ecolor='o', elinewidth=1, marker='p', markerfacecolor='o', markeredgecolor='o', linewidth=2, label='photon count rate')
ax.set_ylabel('Count rate [$Hz$]')
t = np.linspace(xlim[0], xlim[1])
ax.semilogy(t, photon_rate(t, eta), ls='--', c='k', label='Fit Eq. (3)')
ax.axhline(Nph_dark, ls='-.', c='k', label='dark count rate')
ax.set_ylim(ylim)
ax.set_xlim(xlim)
ax.set_xlabel('Radiator Temperature [K]')
ax.legend(handles=ax.get_legend_handles_labels()[0][::-1], 
          labels=ax.get_legend_handles_labels()[1][::-1], loc='upper left', ncols=3, handlelength=1.5, columnspacing=0.25)

plt.savefig('figures/%s_efficiency.pdf' % name)
plt.show()

