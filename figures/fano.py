import scipy.constants as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlibcolors
plt.style.use('figures/matplotlibrc')

def fano(wl, Tc, J=0):
    Eph = sc.h * sc.c / wl
    F = 0.2
    Delta = 1.76 * sc.k * Tc
    eta_pb = 0.59
    return 1/(2*np.sqrt(2*np.log(2))) * np.sqrt(eta_pb * Eph / (Delta * (F+J)))


fig, ax = plt.subplots(constrained_layout=True, figsize=(6,3))
wl = np.linspace(0.1, 30, 1000)*1e-6
Tc_Al = 1.25
Rfano = fano(wl, Tc_Al)
devisser_mebrane = np.array([[1545, 986, 673, 402], [19,29,40,51]])
devisser_substrate = np.array([[1545, 986, 673, 402], [10,13,18,21]])
ras_hist = np.array([[3.8, 8.5, 18.5, 25], [5.7,4.9,1.8, 2.9]])
ras_hist = np.array([[3.8, 8.5, 18.5, 25], [5.7,4.9,1.8, 2.9]])
ax.plot(wl*1e6, Rfano, label='Fano limit')
ax.plot(wl*1e6, fano(wl, Tc_Al, J=3.1), label='$R_{\mathrm{phonon}}$, J=3.1')
ax.plot(wl*1e6, fano(wl, Tc_Al, J=.38), label='$R_{\mathrm{phonon}}$, J=0.38')
ax.scatter(25, 2.92, label='Day et al. 2025, substrate', c='o')
ax.scatter(devisser_mebrane[0]*1e-3, devisser_mebrane[1], label='De Visser et al. 2021, membrane', c='y', marker='s')
ax.scatter(devisser_substrate[0]*1e-3, devisser_substrate[1], label='De Visser et al. 2021, substrate', c='o', marker='s')
ax.scatter(ras_hist[0], ras_hist[1], label='This work', facecolor='None', edgecolor='y', linewidth=2)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('$R_{Fano}$ [-]')
ax.set_xlim([.1,30])
ax.set_ylim([1,200])
ax.set_xscale('log')    
ax.set_yscale('log')    
ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
        ncols=2, mode="expand", borderaxespad=0., fontsize=9)
plt.show()