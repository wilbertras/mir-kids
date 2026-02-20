"""
initialize_filterstacks.py

This module initializes the filter stacks for the configurations '3.8um', '8.5um', '18.5um', and '25um'. 
It loads various optical filters and calculates the total tranmission based on the filter 
configurations. The total spectral radiance is calculated by multiplying with a planck spectrum at specified temperatures. The results are stored in 
a dictionary and saved to a pickle file for later use.

Two figures are generated: 
- all transmission curves of all the individual filters
- total spectral radiances for the setup configurations '3.8um', '8.5um', '18.5um', and '25um'.

"""
#--------------------------------------------------
# Import modules
# -------------------------------------------------
from scipy.signal import savgol_filter
import scipy.constants as sc
import matplotlib.pyplot as plt
import numpy as np
import pickle

from . import functions as ft
from .initialize_filters import load_all_filters

# from .utils import matplotlibcolors
# plt.style.use('./utils/matplotlibrc')


def initialize_filterstacks(path2data, name):
    #--------------------------------------------------
    # Load kid_dict
    # -------------------------------------------------
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)
    print("Loaded file %s" % path2kid_dict)


    keys = ['3.8um', '8.5um', '18.5um', '25um']

    #--------------------------------------------------
    # Load and initiate filters
    # -------------------------------------------------
    filters = load_all_filters()

    wl = filters['wl']
    bp38 = filters['bp38']
    bp85 = filters['bp85']
    bp185 = filters['bp185']
    caf2 = filters['caf2']
    znse = filters['znse']
    nd1 = filters['nd1']
    nd2 = filters['nd2']
    nd3 = filters['nd3']
    ger = filters['ger'] 
    bp25 = filters['bp25']
    sp_a = filters['sp_a']
    sp_b = filters['sp_b']
    lp = filters['lp']
    si = filters['si']

    #--------------------------------------------------
    # Compute the relevant blackbody radiation for the different configurations.
    # -------------------------------------------------
    Tbb38 = 293
    Tbb85 = 293
    Tbb185 = 160
    Tbb25 = 24
    planck38 = ft.planck_wl(wl, Tbb38)*1e-6
    planck85 = ft.planck_wl(wl, Tbb85)*1e-6
    planck185 = ft.planck_wl(wl, Tbb185)*1e-6
    planck25 = ft.planck_wl(wl, Tbb25)*1e-6

    #--------------------------------------------------
    # Setup specifics
    # -------------------------------------------------
    omegaBF = np.pi*(10e-3)**2/(300e-3)**2  # opening angle BF cryostat (3.8um, 8.5um, 18.5um)
    omegaADR = np.pi*(10e-3)**2/(65e-3)**2  # opening angle ADR cryostat (25um)
    A = .25 * np.pi*(1.55e-3)**2                           # area of KID

    #--------------------------------------------------
    # Compute full stack transmissions
    # -------------------------------------------------
    theta38 = bp38**4 * caf2**5 * nd2 * nd3 * si * A * omegaBF
    theta85 = bp85**4 * caf2**5 * nd2 * nd3 * si * A * omegaBF
    theta185 = bp185**5 * nd1 * nd3 * znse**2 * si * A * omegaBF
    theta25 = bp25**3 * lp * sp_a**3 * sp_b * si * A * omegaADR

    #--------------------------------------------------
    # Compute spectral radiance
    # -------------------------------------------------
    tot38 = theta38 * planck38
    tot85 = theta85 * planck85
    tot185 = theta185 * planck185
    tot25 = theta25 * planck25

    #--------------------------------------------------
    # Save tranmission data
    # -------------------------------------------------
    thetas = [theta38, theta85, theta185, theta25]
    for i, key in enumerate(keys):
        kid_dict[name][key]['theta'] = [wl, thetas[i]]
    with open(path2kid_dict, 'wb') as f:
        pickle.dump(kid_dict, f)
    print("Updated file %s" % path2kid_dict)


    #--------------------------------------------------
    # Plotting all individual filters
    # -------------------------------------------------
    wl *= 1e6
    fig, axes = plt.subplot_mosaic('a;b;c;d', figsize=(18.5/2/2.54,12/2.54), constrained_layout=True)
    xticks_minor = np.hstack((np.arange(1,11, 1), np.arange(20,110,10)))
    xticks_major = [1, 2, 5, 10, 20, 50, 100]
    xlim = [1, 100]
    ax = axes['a']
    band = wl < 45
    ax.semilogx(wl[band], bp38[band], label='BP$_{3.8}$', c='b')
    ax.semilogx(wl[~band], bp38[~band], c='b', ls=':')
    # ax.fill_between(wl, bp38, 0, where=(wl < 45), alpha=0.3, color='b')
    ax.semilogx(wl[band], bp85[band], label='BP$_{8.5}$', c='y')
    ax.semilogx(wl[~band], bp85[~band], c='y', ls=':')
    # ax.fill_between(wl, bp85, 0, where=(wl < 45), alpha=0.3, color='y')
    band = wl < 40
    ax.semilogx(wl[band], bp185[band], label='BP$_{18.5}$', c='o')
    ax.semilogx(wl[~band], bp185[~band], c='o', ls=':')
    # ax.fill_between(wl, bp185, 0, where=(wl < 40), alpha=0.3, color='o')
    ax.legend(ncols=3, loc='upper right', handlelength=1, columnspacing=0.5)


    ax = axes['b']
    band = wl < 25
    ax.semilogx(wl[band], znse[band], label='ZnSe', c='b')
    ax.semilogx(wl[~band], znse[~band], c='b', ls=':')
    # ax.fill_between(wl, znse, 0, where=(wl < 25), alpha=0.3)
    band = wl < 16
    ax.semilogx(wl[band], caf2[band], label='$CaF_2$', c='y')
    ax.semilogx(wl[~band], caf2[~band], c='y', ls=':')
    ax.semilogx(wl, si, label='Si', c='o')
    ax.legend(ncols=3, loc='upper right', handlelength=1, columnspacing=0.5)
    # ax.fill_between(wl, caf2, 0, where=(wl < 16), alpha=0.3)

    ax = axes['c']
    c='p' 
    upband = (wl > 1.67)
    lowband = (wl < 77)
    ax.semilogx(wl, bp25, label='BP$_{25}$', zorder=3, c=c)
    ax.semilogx(wl[~upband], bp25[~upband], zorder=0, ls=':', c=c)
    ax.semilogx(wl[~lowband], bp25[~lowband], zorder=0, ls=':', c=c)
    # ax.fill_between(wl, bp25, 0, where=((wl < 77) & (wl > 1.67)), alpha=0.3)
    c='b' 
    ax.semilogx(wl, sp_a, label='SP$_{A}$', zorder=2, c=c)
    ax.semilogx(wl[~upband], sp_a[~upband], zorder=0, ls=':', c=c)
    ax.semilogx(wl[~lowband], sp_a[~lowband], zorder=0, ls=':', c=c)
    # ax.fill_between(wl, sp_a, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
    c='y' 
    ax.semilogx(wl, sp_b, label='SP$_{B}$', zorder=1, c=c)
    ax.semilogx(wl[~upband], sp_b[~upband], zorder=0, ls=':', c=c)
    ax.semilogx(wl[~lowband], sp_b[~lowband], zorder=0, ls=':', c=c)
    # ax.fill_between(wl, sp_b, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
    c='o' 
    upband = (wl > 1.67)
    lowband = (wl < 82)
    ax.semilogx(wl[upband&lowband], lp[upband&lowband], label='LP', zorder=0, c=c)
    ax.semilogx(wl[~upband], lp[~upband], zorder=0, ls=':', c=c)
    ax.semilogx(wl[~lowband], lp[~lowband], zorder=0, ls=':', c=c)
    # ax.fill_between(wl, lp, 0, where=((wl < 82)&(wl > 1.67)), alpha=0.3)
    ax.legend(ncols=2, loc='upper left', handlelength=1, columnspacing=0.5)

    ax = axes['d']
    c='b'
    band = wl < 18
    ax.loglog(wl[band], nd1[band], label='ND$_{1}$', c=c)
    ax.semilogx(wl[~band], nd1[~band], c=c, ls=':')
    # ax.fill_between(wl, nd1, 0, where=(wl < 18), alpha=0.3)
    c='y'
    ax.loglog(wl[band], nd2[band], label='ND$_{2}$', c=c)
    ax.semilogx(wl[~band], nd2[~band], c=c, ls=':')
    # ax.fill_between(wl, nd2, 0, where=(wl < 18), alpha=0.3)
    c='o'
    ax.loglog(wl[band], nd3[band], label='ND$_{3}$', c=c)
    ax.semilogx(wl[~band], nd3[~band], c=c, ls=':')
    # ax.fill_between(wl, nd3, 0, where=(wl < 18), alpha=0.3)

    ax.grid(True, which='major', axis='y')
    ax.grid(False, which='minor', axis='y')
    ax.grid(True, which='both', axis='x')
    ax.set_xlabel('Wavelength [µm]')
    # ax.set_ylabel('Transmission [-]')
    ax.legend(ncols=3, loc='upper right', handlelength=1, columnspacing=0.5)

    for i, id in enumerate(axes):
        ax = axes[id]
        if i < 3:
            ylim = [0,1]
            ax.set_ylim(ylim)
            yticks = np.linspace(ylim[0], ylim[1], 5, endpoint=True)
            ax.set_yticks(yticks, minor=True)
            ax.set_yticks(yticks, minor=False)
            ax.set_xticklabels(yticks, minor=False)
        else:
            ylim = [-4, 0]
            ax.set_ylim([10**ylim[0], 10**ylim[1]])
            yticks = np.logspace(ylim[0], ylim[1], 5, endpoint=True)
            ax.set_yticks(yticks, minor=False)
        ax.set_xlim(xlim)
        ax.set_ylabel('Transmission')
        ax.set_xticks(xticks_major, minor=False)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(xticks_major, minor=False)

    # plt.savefig('figures/%s_filters.pdf' % name)


    #--------------------------------------------------
    # Plot spectral radiance
    # -------------------------------------------------

    fig, axes = plt.subplot_mosaic('a', figsize=(18.5/2.18/2.54, 9/2.54), constrained_layout=True, sharey=True, sharex=True)
    ax = axes['a']
    ax.plot(wl, savgol_filter(tot38, 21, 1), label='3.8 µm; $T_\mathrm{lab}=%d\ K$' % Tbb38, color='b')
    ax.loglog(wl, savgol_filter(tot85, 21, 1), label='8.5 µm; $T_\mathrm{lab}=%d\ K$' % Tbb85, color='y', ls='--')
    ax.loglog(wl, savgol_filter(tot185, 21, 1), label='18.5 µm; $T_{bb}=%d\ K$' % Tbb185, color='o', ls='--')
    ax.loglog(wl, savgol_filter(tot25, 21, 1), label='25 µm; $T_{bb}=%d\ K$' % Tbb25, color='p', ls='--')
    ax.set_ylim([1e-28, 1e-12])
    ax.set_xlim([1,100])
    ax.set_xlabel('Wavelength [µm]')
    ax.set_ylabel('Spectral radiance [$W sr^{-1}m^{-2}\mu m^{-1}$]')
    ax.legend(loc='upper center', ncols=2, mode="expand", borderaxespad=0., frameon=False, handlelength=1, columnspacing=0.25)

    plt.savefig('figures/%s_radiance.pdf' % name)
    plt.show()


if __name__ == "__main__":
    initialize_filterstacks()