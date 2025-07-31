import scipy.constants as sc
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlibcolors
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.ticker as mticker
plt.style.use('figures/matplotlibrc')

def fano(wl, Tc, J=0):
    Eph = sc.h * sc.c / wl
    F = 0.2
    Delta = 1.76 * sc.k * Tc
    eta_pb = 0.59
    return 1/(2*np.sqrt(2*np.log(2))) * np.sqrt(eta_pb * Eph / (Delta * (F+J)))

[wl_log, tot25] = np.load('figures/25um_filterstack.npy')
wl = np.arange(20, 30, 0.1)
theta = interp1d(wl_log, tot25)(wl)
half_max = np.amax(theta)/2
peak_idx = int(len(theta)/2)
left_index = np.where(theta[:peak_idx] < half_max)[0][-1]
right_index = np.where(theta[peak_idx:] < half_max)[0][0] + peak_idx
fwhm = wl[right_index] - wl[left_index]
mu = (wl[left_index] + wl[right_index]) / 2
R_filter = mu/fwhm
print(R_filter)
# fig, ax = plt.subplots()
# ax.plot(wl, theta)
# ax.axvline(wl[left_index], c='k')
# ax.axvline(wl[right_index], c='k')
def oneover(x, a):
    return a / x

fig, ax = plt.subplots(constrained_layout=True, figsize=(9.25/2.54,8/2.54))
wl = np.linspace(3, 30, 1000)
wls = np.array([3.8, 8.5, 18.5, 25])
Tc_Al = 1.54
Rfano = fano(wl*1e-6, Tc_Al)
Rfano_wls = fano(wls*1e-6, Tc_Al, J=.38)
R = np.array([6.04,5.56,2.77,2.45])
Rsn = np.array([15.68,12.77,3.81,4.27])
tqp = np.array([204, 243, 272, 313])
filter_idx = np.array([0,0,0,1])
R = np.sqrt(1/(1/R**2 - filter_idx/R_filter**2))
R_exp = np.sqrt(1/(1/R**2 - 1/Rsn**2))
print(R)

popt, pcov = curve_fit(oneover, wls, Rsn)
print(popt)

ax.plot(wl, fano(wl*1e-6, Tc_Al, J=.38), label='$R_{Fano}$, $J=0.38$', c='k', linestyle='-', zorder=-4)
ax.plot(wl, fano(wl*1e-6, Tc_Al, J=3.1), label='$R_{Fano}$, $J=3.1$', c='k', linestyle='--', zorder=-4)
ax.plot(wl, Rsn[1]*wls[1]/wl, label='$R_{sn}/\lambda$', c='k', linestyle='-.', zorder=-4)
ax.scatter(wls, R, label='$R$', facecolor='None', edgecolor='b', linewidth=2, marker='s', zorder=-1)
ax.scatter(wls, Rsn, label='$R_{sn}$', facecolor='None', edgecolor='y', linewidth=2, marker='o', zorder=-2)
ax.scatter(wls, R_exp, label='$R_{i}$', facecolor='None', edgecolor='o', linewidth=2, marker='^', zorder=-3)
ax.scatter(25, 2.92, label='Day(2024)', marker='D', linewidth=2, facecolor='p', edgecolor='p', zorder=-3)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Energy resolution [-]')
ax.set_xlim([3,30])
ax.set_ylim([2,20])

ax.set_xscale('log')    
ax.set_yscale('log')    
ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
        ncols=3, mode="expand", borderaxespad=0.)
plt.savefig('figures/fano.pdf')
# fig, ax = plt.subplots(constrained_layout=True, figsize=(9.25/2.54,8/2.54))
# ax.plot(wls, tqp)
plt.show()