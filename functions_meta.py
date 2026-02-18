import numpy as np
import matplotlib.pyplot as plt
import functions as ft
from scipy.signal import convolve, fftconvolve, find_peaks, medfilt
from scipy.signal import welch
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
import matplotlibcolors as matplotlibcolors
plt.style.use('matplotlibrc')


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


def peaks_vs_thresholds(dir, kid, pread, file_type, pw, pw_offset, lifetime, nr_stds, chuncks, chuncksize, filter_type, fit_tqp, exclude_dc=True, outlier_filter=4.5, secondary_filter=.1, binsize=0.025):
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
    for i, nr_std in enumerate(nr_stds):
        ph = nr_std * std
        pp = nr_std * std
        pulses, neg_pulses = get_pulses(pulse_files, ph, pp, pw, pw_offset, chuncksize, chuncks[i], exp_filter)
        pulses_fft = fft(pulses, axis=1)
        neg_pulses_fft = fft(neg_pulses, axis=1)
        H_opt = np.sum((opt_filter*pulses_fft)[:, exclude_dc:], axis=1).real
        neg_H_opt = np.sum((opt_filter*neg_pulses_fft)[:, exclude_dc:], axis=1).real
        H_opts.append(H_opt)
        neg_H_opts.append(neg_H_opt)
        print('\nAnalysed chunck %d out of %d' % (i+1, len(chuncks)), end='\r')

    mask = filter_pulses(pulses, lifetime/2, outlier_filter, secondary_filter)
    pulses = pulses[mask]
    pulses_fft = pulses_fft[mask]
    H_opts[-1] = H_opts[-1][mask]
    Rsn = np.mean(H_opt) / fwhm

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
    colors = ['b', 'y', 'o'] 
    for i, H_opt in enumerate(H_opts):
        _ = ax.hist(H_opt, bins=bins, facecolor=colors[i], alpha=.75, label='%d$\sigma$' % (nr_stds[i]), zorder=i+1)

    max_y = np.amax(np.histogram(H_opts[-1], bins=bins)[0])
    ax.set_ylim([0, max_y*1.5])
    ax.set_xlim([-.25,2])
    ax.set_ylabel('Counts')
    ax.set_xlabel('Pulse heights [rad]')

    H_opt = H_opts[-1]
    c = colors[len(H_opts)-1]

    ax = axes['f']
    mean_pulse = np.mean(pulses, axis=0)    
    std_pulse = np.std(pulses, axis=0)
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
    R_opt, pdf_y, pdf_x, _, _ = ft.resolving_power(H_opt, binsize)
    _, _, _, _, noise_fwhm = ft.resolving_power(optimal_noise, binsize)
    R0 = np.mean(H_opt) / noise_fwhm
    ax.plot(pdf_x, pdf_y, c='k', ls='--', label='KDE', lw=1)
    ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
            ncols=4, mode="expand", borderaxespad=0.)
    print('\nR = %.2f, Rsn = %.2f, R0 = %.2f, %d pulses' % (R_opt, Rsn, R0, len(pulses)))
    print('tqp = %d +- %.1f us' % (tqp, perr[1]))

    dict = {}
    dict['pulse template'] = norm_pulse
    dict['pulses'] = pulses
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
    dict['Rsn'] = Rsn
    dict['R0'] = R0
    dict['kde'] = [pdf_x, pdf_y]
    dict['binsize'] = binsize
    dict['nxx'] = [freqs, avg_psd_noises_onesided]
    dict['dxx'] = [freqs, avg_psd_pulses_onesided]
    dict['mxx'] = [freqs, pulse_psd_onesided]
    dict['f0'] = f0
    return dict