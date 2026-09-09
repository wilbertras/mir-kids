from scipy.signal import medfilt, convolve, fftconvolve, find_peaks
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from . import functions as ft
import numpy as np
import pickle
from copy import copy


def get_pulses(pulse_files, ph, pp, pw, pw_offset, chuncksize, nr_chuncks, window):
    nr_files = len(pulse_files)
    if nr_chuncks:
        nr_req_files = np.amin((chuncksize * nr_chuncks, nr_files))
    else:
        nr_req_files = nr_files
    analysed_files = 0
    pulses = []
    single_idx = []
    single_locs = []
    neg_pulses = []
    neg_single_idx = []
    neg_single_locs = []
    while analysed_files < nr_req_files:
        if analysed_files + chuncksize > nr_files:
            chuncksize = nr_files - analysed_files
        amp, phase, _ = ft.get_data(pulse_files[analysed_files:analysed_files+chuncksize])
        signal = ft.coord_transformation(phase, amp, coord='circle', response='phase')
        medsignal = medfilt(signal)
        len_file = int(len(signal) / chuncksize)
        locs, _ = ft.find_pks(medsignal, ph, pp, window)
        neg_locs, _ = ft.find_pks(-medsignal, ph, pp, window)
        pulses_chunck, single_idx_chunck = ft.get_single_pulses(signal, locs, pw, 
                                                                pw_offset)
        neg_pulses_chunck, neg_single_idx_chunck = ft.get_single_pulses(signal, neg_locs, pw, 
                                                                pw_offset)
        if len(pulses_chunck):
            pulses.append(pulses_chunck)
            single_idx.append(single_idx_chunck)
            single_locs.append(locs[single_idx_chunck] + len_file*analysed_files)
        if len(neg_pulses_chunck):
            neg_pulses.append(neg_pulses_chunck)
            neg_single_idx.append(neg_single_idx_chunck)
            neg_single_locs.append(neg_locs[neg_single_idx_chunck] + len_file*analysed_files)
        analysed_files += chuncksize
        print('Analysed %d out of %d files' % (analysed_files, nr_req_files), end='\r')
    single_idx = np.hstack(single_idx)
    pulses = np.vstack(pulses)
    single_locs = np.hstack(single_locs)
    neg_single_idx = np.hstack(neg_single_idx)
    neg_pulses = np.vstack(neg_pulses)
    neg_single_locs = np.hstack(neg_single_locs)
    return pulses, neg_pulses


def filter_pulses(pulses, sw, outlier_filter, secondary_filter):
    sw = int(sw)
    box = np.ones(sw)/sw
    if secondary_filter == 1:
        mask_doubles = np.ones(len(pulses), dtype=bool)
    else:
        mph = np.amax(convolve(np.mean(pulses, axis=0), box , mode='full')) * secondary_filter
        mask_doubles = np.ones(len(pulses), dtype=bool)
        for i, pulse in enumerate(pulses):
            filtered_pulse = convolve(pulse, box, mode='full')
            peaks, _ = find_peaks(filtered_pulse, height=mph, prominence=mph)
            if len(peaks)>1:
                mask_doubles[i] = 0
    mean = np.mean(pulses, axis=0)
    std = np.std(pulses, axis=0)   
    mask_outlier = np.all(np.abs(pulses - mean) < (outlier_filter * std), axis=1)
    mask = mask_doubles & mask_outlier
    print('Secondaries/Outliers/Total = %d/%d/%d' % (np.sum(~mask_doubles), np.sum(~mask_outlier), len(mask)))
    return mask


def peaks_vs_thresholds(dir, kid, pread, file_type, pw, pw_offset, lifetime, nr_stds, chuncks, chuncksize, filter_type, fit_tqp, binsize=0.025, exclude_dc=True, outlier_filter=4.5, secondary_filter=.1):
    pulse_files, info_files = ft.get_files(dir + 'TD_power', kid, pread, type=file_type)
    nr_files = len(pulse_files)

    amp, phase, _ = ft.get_data(pulse_files[:chuncksize])
    signal = ft.coord_transformation(phase, amp, coord='circle', response='phase')

    f0, Q, Qc, Qi, S21_min, dt, T = ft.get_info(info_files[0])
    sf = round(1 / dt)
    sff = sf / 1e6
    sw = round(lifetime * sff)
    exp_filter = ft.get_window(filter_type, sw)
    std = ft.get_sigma(signal, exp_filter)
    std_raw = ft.get_sigma(signal, [])

    noise_mph = nr_stds[0]
    ph = noise_mph*std
    pp = ph
    noise_locs, _ = ft.find_pks(signal, ph, pp, exp_filter)
    medsignal = medfilt(signal)
    filtered_signal = fftconvolve(medsignal, exp_filter, mode='valid')
    noises = ft.get_single_noises(signal, noise_locs, pw+pw_offset)

    pulse_mph = nr_stds[-1]
    ph = pulse_mph*std
    pp = ph
    locs, props = ft.find_pks(medsignal, ph, pp, exp_filter)
    pulses, single_idx = ft.get_single_pulses(signal, locs, pw, pw_offset)
    mask = filter_pulses(pulses, lifetime/2, outlier_filter, secondary_filter)
    pulses = pulses[mask]
    pulse_template = np.mean(pulses, axis=0)

    norm_pulse = pulse_template / np.amax(pulse_template)
    N = len(norm_pulse)
    L = round(N/2+1)
    df = sf/N
    T = 1/(sf*N)
    freqs = np.arange(0, sf/2+df, df)
    norm_fft = fft(norm_pulse)
    pulse_psd = T * np.abs(fft(pulse_template))**2
    pulses_psd = T * np.abs(fft(pulses))**2
    noises_psd = T * np.abs(fft(noises))**2
    avg_psd_noises = np.mean(noises_psd, axis=0)
    avg_psd_pulses = np.mean(pulses_psd, axis=0)
    normalisation = np.sum((np.abs(norm_fft)**2 / avg_psd_noises)[exclude_dc:])
    sigma = np.real(np.sqrt(1 / (T * normalisation)))
    fwhm = 2.355 * sigma

    opt_filter = norm_fft.conj() / avg_psd_noises / normalisation

    def tau_qp(x, a, tqp):
        return a*np.exp(-x/tqp)
    
    print('Optimal filter constructed with %d pulses and %d noise segments' % (pulses.shape[0], noises.shape[0]))
    fig, axes = plt.subplot_mosaic('abc;def', constrained_layout=True, figsize=(10, 5))
    ax = axes['a']
    t = np.arange(0, pw+pw_offset, 1)
    ax.plot(t, pulses.T, alpha=.2, lw=.1, c='y', label='_nolegend_')
    if len(pulses):
        ax.plot(t, pulses[0], lw=.1, c='y', label='%d pulses' % len(pulses))
    ax.plot(t, pulse_template, label='mean pulse')
    ax.set_xlim([t[0], t[-1]])
    ax.set_xlabel('Time [$\mu$s]')
    ax.set_ylabel('$\\theta$ [rad]')
    _ = ax.legend()
    popt, pcov = curve_fit(tau_qp, t[fit_tqp[0]:fit_tqp[1]], pulse_template[fit_tqp[0]:fit_tqp[1]], p0=[1, 200])
    tqp = popt[1]
    x = np.arange(fit_tqp[0], fit_tqp[1])
    ax.plot(x, tau_qp(x, *popt), ls='--', lw=2, label='$\\tau_{qp}$=%d $\mu$s' % tqp, zorder=3)
    ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
            ncols=2, mode="expand", borderaxespad=0.)
    ax_inset = inset_axes(ax, width="50%", height="50%", loc='upper right')
    ax_inset.plot(t, pulse_template, label='mean pulse')
    ax_inset.set_xlim([t[0], t[-1]])
    ax_inset.set_yscale('log')
    ax_inset.semilogy(x, tau_qp(x, *popt), ls='--', lw=2, label='$\\tau_{qp}$=%d $\mu$s' % tqp, zorder=3)
    ax = axes['b']
    ax.hist(filtered_signal, bins=np.arange(-10*std, 20*std, 0.25*std))
    ax.axvline(noise_mph*std, c='r', lw=1, label='Noise mph=%d$\\sigma$' % noise_mph)
    ax.set_yscale('log')
    xticks = np.arange(-10, 21, 1)
    ax.set_xticks(xticks*std)
    ax.set_xticklabels(xticks)
    ax.set_xlabel('sigma')

    H_opts = []
    neg_H_opts = []
    R_sns = []
    R_opts = []
    pdfs = []
    for i, nr_std in enumerate(nr_stds[::-1]):
        ph = nr_std * std
        pp = nr_std * std
        pulses, neg_pulses = get_pulses(pulse_files, ph, pp, pw, pw_offset, chuncksize, chuncks[i], exp_filter)
        pulses_fft = fft(pulses, axis=1)
        neg_pulses_fft = fft(neg_pulses, axis=1)
        H_opt = np.sum((opt_filter*pulses_fft)[:, exclude_dc:], axis=1).real
        neg_H_opt = np.sum((opt_filter*neg_pulses_fft)[:, exclude_dc:], axis=1).real
        mask = filter_pulses(pulses, lifetime/2, outlier_filter, secondary_filter)
        pulses = pulses[mask]
        if i==0:
            pulses_opt = copy(pulses)
        if i < len(nr_stds)-1: 
            H_opt = H_opt[mask]  
            R_opt, pdf_y, pdf_x, _, _ = ft.resolving_power(H_opt, binsize)
            R_sn = np.mean(H_opt) / fwhm
            R_sns.append(R_sn)
            R_opts.append(R_opt)
            pdfs.append([pdf_x, pdf_y])
        else:
            pass
        H_opts.append(H_opt)
        neg_H_opts.append(neg_H_opt)
        print('\nAnalysed chunck %d out of %d' % (i+1, len(chuncks)), end='\r')  
    H_opts = H_opts[::-1]
    neg_H_opts = neg_H_opts[::-1]
    R_sns = R_sns[::-1]
    R_opts = R_opts[::-1]
    pdfs = pdfs[::-1]

    H_opt = H_opts[-1]
    R_opt = R_opts[-1]
    R_sn = R_sns[-1]
    pdf = pdfs[-1]

    ax = axes['c']
    opt_filter_td = ifft(opt_filter[exclude_dc:]).real
    ax.plot(opt_filter_td)
    ax = axes['d']
    pulse_psd_onesided = pulse_psd[:L]
    pulse_psd_onesided[1:L-1] *= 2
    avg_psd_pulses_onesided = avg_psd_pulses[:L]
    avg_psd_pulses_onesided[1:L-1] *= 2
    avg_psd_noises_onesided = avg_psd_noises[:L]
    avg_psd_noises_onesided[1:L-1] *= 2
    ax.semilogx(freqs, 10*np.log10(pulse_psd_onesided), label='Single Pulse')
    ax.semilogx(freqs, 10*np.log10(avg_psd_pulses_onesided), label='Average Pulse')
    ax.semilogx(freqs, 10*np.log10(avg_psd_noises_onesided), label='Average Noise')
    ax = axes['e']
    bins=np.arange(-.5, 2, binsize)
    # noises = np.array([signal[i:i+pw+pw_offset] for i in range(0, len(signal)-(pw+pw_offset), pw+pw_offset)])
    noises_fft = fft(noises, axis=1)
    optimal_noise = np.sum((opt_filter*noises_fft)[:, exclude_dc:], axis=1).real
    ax.hist(optimal_noise, bins=bins, facecolor='gray', label='Noise', zorder=0)
    # ax.hist(-neg_H_opts[0], bins=bins, facecolor='r', alpha=.5, label='Neg. Pulses', zorder=1)
    print('Nr of neg pulses at lowest threshold: %d' % len(neg_H_opts[0]))
    # ax.set_yscale('log')
    colors = ['b', 'y', 'o', 'p'] 
    for i, H in enumerate(H_opts):
        if i<1:
            label = '%d$\sigma$' % (nr_stds[i])
        else:
            label = '%d$\sigma$, R=%.1f' % (nr_stds[i], R_opts[i-1])
        _ = ax.hist(H, bins=bins, facecolor=colors[i], alpha=.75, label=label, zorder=i+1)

    max_y = np.amax(np.histogram(H_opt, bins=bins)[0])
    ax.set_ylim([0, max_y*1.5])
    ax.set_xlim([-.25,2])
    ax.set_ylabel('Counts')
    ax.set_xlabel('Pulse heights [rad]')

    c = colors[len(H_opts)-1]

    ax = axes['f']
    mean_pulse = np.mean(pulses_opt, axis=0)    
    std_pulse = np.std(pulses_opt, axis=0)
    ax.plot(t, mean_pulse, c=c, label='Mean pulse main')
    ax.fill_between(t, mean_pulse - std_pulse, mean_pulse + std_pulse, color=c, alpha=.2, label='$\pm1$ std')
    popt, pcov = curve_fit(tau_qp, t[fit_tqp[0]:fit_tqp[1]], mean_pulse[fit_tqp[0]:fit_tqp[1]], p0=[1, 200])
    tqp = popt[1]
    perr = np.sqrt(np.diag(pcov))
    y = tau_qp(x, *popt)
    ax.plot(x, y, c='k', ls='--', lw=1.5, label='fit $\\tau_{qp}$', zorder=3)
    xticks = np.arange(t[0], t[-1], 250)
    ax_inset = inset_axes(ax, width="50%", height="50%", loc='upper right')
    ax_inset.semilogy(t, mean_pulse, c=c, label='mean pulse')
    ax_inset.semilogy(x, y, c='k', ls='--', lw=1, label='fit $\\tau_{qp}$', zorder=3)
    ax_inset.set_xlim([t[0], t[-1]])
    ax_inset.set_ylim([1e-3, 2])
    ax_inset.set_xticks(xticks)
    ax_inset.set_xticklabels(xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([-.25, 2])
    ax.set_ylabel('Pulse heights [rad]')
    ax.set_xlabel('Time [$\mu$s]')
    ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',ncols=3, mode="expand", borderaxespad=0., handlelength=1.5)

    ax = axes['e']
    _, _, _, _, noise_fwhm = ft.resolving_power(optimal_noise, binsize)
    R0 = np.mean(H_opt) / noise_fwhm
    ax.plot(pdf[0], pdf[1], c='k', ls='--', label='KDE', lw=1)
    ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
            ncols=4, mode="expand", borderaxespad=0.)
    
    dR_opt = [R_opts[-3]-R_opt, R_opts[-2]-R_opt]
    dR_sn = [R_sns[-3]-R_sn, R_sns[-2]-R_sn]

    print('\nR = %.2f(%.2f:+%.2f), Rsn = %.2f(%.2f:+%.2f), R0 = %.2f, %d pulses' % (R_opt, dR_opt[0], dR_opt[1], R_sn, dR_sn[0], dR_sn[1], R0, len(pulses_opt)))
    print('tqp = %d +- %.1f us' % (tqp, perr[1]))

    dict = {}
    dict['pulse template'] = norm_pulse
    dict['pulses'] = pulses_opt
    dict['opt_filter'] = opt_filter
    dict['Hopts'] = H_opts
    dict['Hopt noise'] = optimal_noise
    dict['stds'] = nr_stds
    dict['std'] = std
    dict['std raw'] = std_raw
    dict['tqp'] = tqp
    dict['fit'] = [x, y]
    dict['kde'] = [pdf_x, pdf_y]
    dict['Ropt'] = R_opt
    dict['dRopt'] = [R_opts[-3]-R_opt, R_opts[-2]-R_opt]
    dict['Rsn'] = R_sn
    dict['dRsn'] = [R_sns[-3]-R_sn, R_sns[-2]-R_sn]
    dict['R0'] = R0
    dict['dR0'] = [R0-R_sns[-2], R0-R_sns[-3]]
    dict['kde'] = pdf
    dict['binsize'] = binsize
    dict['nxx'] = [freqs, avg_psd_noises_onesided]
    dict['dxx'] = [freqs, avg_psd_pulses_onesided]
    dict['mxx'] = [freqs, pulse_psd_onesided]
    dict['f0'] = f0
    return dict


def pulse_height_distribution(path2data, name, wl, thresholds, chuncks, lifetime, fit_tqp, secondary_filter, binsize, pw=375, pw_offset=25, chuncksize=50, filter_type='exp', file_type='vis'):
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)
    print("Loaded file %s" % path2kid_dict)

    dir = kid_dict[name][wl]['dir']
    kid = kid_dict[name][wl]['kid']
    pread = kid_dict[name][wl]['pread']

    dict = peaks_vs_thresholds(dir, kid, pread, file_type, pw, pw_offset, lifetime, thresholds, 
                                chuncks, chuncksize, filter_type, fit_tqp, binsize=binsize, secondary_filter=secondary_filter)
    
    kid_dict[name][wl].update(dict)
    with open(path2kid_dict, 'wb') as f:
        pickle.dump(kid_dict, f)
    print("Updated file %s" % path2kid_dict)


def plot_distributions(path2data, name):
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)
            
    colors = ['b', 'y', 'o', 'p']
    wls = ['3.8um', '8.5um', '18.5um', '25um']
    Q_bf_dark = kid_dict[name]['BF dark']['Q']
    xlims = [1.75, 1.5, 1, 1]

    fig, axes = plt.subplot_mosaic('aaabbb;ccddee', constrained_layout=True, figsize=(18.5/2.54, 10/2.54))
    axs = 'acde'
    for i, key in enumerate(wls):
        scale = Q_bf_dark / kid_dict[name][key]['Q']
        color = colors[i]
        item = kid_dict[name][key]
        ax = axes[axs[i]]
        binsize = item['binsize']
        H_noise = item['Hopt noise']
        H_opts = item['Hopts']
        nr_stds = item['stds']
        bins = np.arange(-2, 3, binsize*scale)
        _ = ax.hist(H_noise*scale, bins=bins, facecolor='gray', label='Noise', zorder=0)
        _ = ax.hist(H_opts[0]*scale, bins=bins, facecolor=color, alpha=.75, label='%d$\sigma$' % (nr_stds[0]), zorder=1)
        [kdex, kdey] = item['kde']
        # kdex *= scale
        ax.plot(kdex*scale, kdey, c='k', ls='--', label='KDE (%d$\sigma$)' % (nr_stds[-1]), lw=1,  zorder=3)
        ax.set_ylim([0, np.amax(kdey)*1.3])
        ax.set_xlim([-.25, xlims[i]])
        ax.set_ylabel('Counts')
        ax.set_xlabel('Pulse height [rad]')
        ax.legend(bbox_to_anchor=(0., .9, 1., .102), loc='upper left',
                                    ncols=4, mode="expand", borderaxespad=0., handlelength=1)
        title = '%s $\mu m$' % wls[i][:-2]
        if i:
            ax.set_title(title)
        else:
            fig.suptitle(title, fontsize=10)
    ax = axes['a']
    ax.annotate('far-IR', xy=(0.3, 800), xytext=(0.5, 900),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('indirect', xy=(.6, 200), xytext=(.7, 500),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate('direct', xy=(1.46,740), xytext=(1, 850),
                arrowprops=dict(facecolor='black', shrink=0.05))
    ax = axes['b']
    wl = '3.8um'
    c='b'
    scale = Q_bf_dark / kid_dict[name][wl]['Q']
    pulses = kid_dict[name][wl]['pulses']
    mean_pulse = np.mean(pulses, axis=0)*scale 
    std_pulse = np.std(pulses, axis=0)*scale
    t = np.arange(len(mean_pulse))
    ax.plot(t, mean_pulse, c=c, label='Mean pulse')
    ax.fill_between(t, mean_pulse - std_pulse, mean_pulse + std_pulse, color=c, alpha=.2, label='$\pm1$ std')
    [x, y] = kid_dict[name][wl]['fit']
    y *= scale
    ax.plot(x, y, c='k', ls='--', lw=1.5, label='fit $\\tau_{qp}$', zorder=3)
    xticks = np.linspace(t[0], t[-1]+1, 9, endpoint=True, dtype=int)    
    ax_inset = inset_axes(ax, width="45%", height="45%", loc='center right')
    ax_inset.semilogy(t, mean_pulse, c=c, label='mean pulse')
    ax_inset.semilogy(x, y, c='k', ls='--', lw=1, label='fit $\\tau_{qp}$', zorder=3)
    ax_inset.set_xlim([t[0], t[-1]+1])
    ax_inset.set_ylim([1e-3, 2])
    ax_inset.set_xticks(xticks[:-1:2])
    ax_inset.set_xticklabels(xticks[:-1:2])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xlim([t[0], t[-1]+1])
    ax.set_ylim([-.25, 2])
    ax.set_ylabel('Pulse heights [rad]')
    ax.set_xlabel('Time [$\mu$s]')
    ax.legend(bbox_to_anchor=(0., .9, 1., .102), loc='upper left',ncols=3, mode="expand", borderaxespad=0., handlelength=2)

    plt.savefig('figures/%s_pulse_height_distributions.pdf' % name)


def plot_resolving_powers(path2data, name):
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)

    wls = ['3.8um', '8.5um', '18.5um', '25um']

    fig, axes = plt.subplot_mosaic('a', figsize=(18.5/2/2.54, 7/2.54), constrained_layout=True)
    ax = axes['a']
    Ropts = []
    R0s = []
    Reffs = []
    Rsns = []
    Ris = []
    Rfilters = [83.47,83.47,83.55,13.48]
    # Ndarks = []
    lambdas = []
    dRs = []
    dRsns = []
    for wl in wls:
        lambdas.append(float(wl[:-2]))
        item = kid_dict[name][wl]
        Ropts.append(item['Ropt'])
        Rsns.append(item['Rsn'])
        R0s.append(item['R0'])
        dRs.append(item['dRopt'])
        dRsns.append(item['dRsn'])
        # Ndarks.append([item['dcr']['3'], item['dcr']['4'], item['dcr']['5']])
    lambdas = np.array(lambdas)
    Reffs = np.sqrt(1/(1/np.asarray(Ropts)**2 - 1/np.asarray(Rfilters)**2))
    Ris = np.sqrt(1/(1/np.asarray(Ropts)**2 - 1/np.asarray(Rsns)**2))
    dRs = np.abs(np.array(dRs).reshape(len(dRs), 2).T)
    dRsns = np.abs(np.array(dRsns).reshape(len(dRsns), 2).T)
    dRis_plus = np.sqrt(1/(1/np.asarray(Reffs+dRs[1])**2 - 1/np.asarray(Rsns+dRsns[1])**2)) - Ris
    dRis_min = np.sqrt(1/(1/np.asarray(Reffs-dRs[0])**2 - 1/np.asarray(Rsns-dRsns[0])**2)) - Ris
    dRis = np.abs(np.vstack((dRis_min, dRis_plus)))
    print(dRs)
    print(dRsns)
    print(dRis)
    # Ndarks = np.array(Ndarks)

    ax.errorbar(lambdas, Ropts, xerr=None, yerr=dRs, label='$R$', marker='o', color='b', markerfacecolor='b', markeredgecolor='None', zorder=1, capsize=2, ms=4, ecolor='b')
    # ax.scatter(lambdas, Reffs, label='$R$', facecolor='None', edgecolor='b', linewidth=2, marker='s', zorder=1)
    ax.errorbar(lambdas, Rsns, xerr=None, yerr=dRsns, label='$R_{SN}$', marker='s', color='y', markerfacecolor='y', markeredgecolor='None', zorder=1, capsize=2, ms=4, ecolor='y')
    # ax.scatter(lambdas, Rsns, label='$R_{SN}$', marker='s', facecolor='y', edgecolor='None', zorder=2, s=40)
    # ax.scatter(wls, Rsns, label='$R_0$', facecolor='y', edgecolor='y', linewidth=2, marker='o', zorder=-2)
    ax.errorbar(lambdas, Ris, xerr=None, yerr=dRis, label='$R_{i}$', marker='^', color='o', markerfacecolor='o', markeredgecolor='None', zorder=1, capsize=2, ms=4, ecolor='o')
    # ax.scatter(lambdas, Ris, label='$R_{i}$', marker='^', facecolor='o', edgecolor='None', zorder=1, s=40)
    ax.scatter(25, 1.89, label='$R$ [23]', marker='o', facecolor='p', edgecolor='None', zorder=2, s=30)
    ax.scatter(25, 2.92, label='$R_i$ [23]', marker='^', facecolor='p', edgecolor='None', zorder=2, s=40)
    wl = np.linspace(3, 30, 1000)
    ax.plot(wl, kid_dict[name]['25um']['Rsn']*25/wl, label='$R_{SN}^{25\mu m}/\lambda$', c='y', linestyle='-.', zorder=0)
    Tc_Al = 1.54
    Rfano = ft.fano(wl*1e-6, Tc_Al)
    Rfano_wls = ft.fano(lambdas*1e-6, Tc_Al, J=.38)
    ax.plot(wl, ft.fano(wl*1e-6, Tc_Al, J=3.1), label='$R^{substrate}_{phonon}$', c='k', linestyle=':', zorder=0)
    ax.plot(wl, ft.fano(wl*1e-6, Tc_Al, J=.38), label='$R^{membrane}_{phonon}$', c='k', linestyle='--', zorder=0)
    ax.plot(wl, ft.fano(wl*1e-6, Tc_Al, J=0), label='$R_{Fano}$', c='k', linestyle='-', zorder=0)
    ax.set_ylabel('Resolving power [-]')
    ax.set_yscale('log')
    ax.set_xscale('log')
    xticks = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    yticks = [1, 2, 5, 10, 20, 50]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    ax.set_xlim([3,30])
    ax.set_ylim([1,50])
    handles, labels = ax.get_legend_handles_labels()
    order = [6, 7, 8, 3, 4, 5, 2, 0, 1]  # Adjust this list to your desired order
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    ax.legend(handles, labels, bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
            ncols=3, mode="expand", borderaxespad=0.)
    print('Ropts= ', Ropts)
    print('Reff = ', Reffs)
    print('R0s = ', R0s)
    print('Ris = ', Ris)
    print('Rsns = ', Rsns)
    # print('Ndarks = \n', Ndarks)
    ax.set_xlabel('Wavelength [µm]')

    plt.savefig('figures/%s_resolving_powers.pdf' % name)




