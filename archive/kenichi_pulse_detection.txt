import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, fftconvolve


def bin2mat(file_path):
    data = np.fromfile(file_path, dtype='>f8', count=-1)
    data = data.reshape((-1, 2))

    I = data[:, 0]
    Q = data[:, 1]

    # From I and Q data to Radius/Magnitude and Phase
    r = np.sqrt(I**2 + Q**2)
    I /= np.mean(r) # Normalize I to 1
    Q /= np.mean(r) # Normalize Q to 1
    R = np.sqrt(I**2 + Q**2)

    P = np.arctan2(Q, I) 
    P = np.pi - P % (2 * np.pi) # Convert phase to be taken from the negative I axis
    return R, P


def get_window(type, tau):
    if type == 'box':
        # Boxcar window, return a rectangular window of length tau/2
        M = int(tau / 2)
        y = windows.boxcar(M, sym=False)
        y /= np.sum(y)
    elif type == 'exp':
        # Exponential window, return an exponential window with decay time tau with length 3*tau
        M = int(tau*3)
        y = windows.exponential(M, center=0, tau=tau, sym=False)
        y /= np.sum(y) 
    elif type == 'None':
        y = []
    else:
        raise Exception('Windowtype was given as %s. Please input a correct window type: "exp", "box" or "None"' % type)
    return y[::-1]      # flip window for convolution


def get_sigma(signal, window):
    # Obtain noise sigma from negative timestream data
    if len(window):    
        signal = fftconvolve(signal, window, mode='valid')
    neg_signal = signal[signal<=0]
    std = np.std(np.hstack((neg_signal, np.absolute(neg_signal))))
    return np.round(std, decimals=3)


def find_pks(signal, ph, pp, window):
    # Smooth timestream data for peak finding
    if len(window):    
        signal = fftconvolve(signal, window, mode='valid')
        window_offset = int(np.argmax(window[::-1]))
    else:
        signal = signal
        window_offset = 0

    # Find peaks in data
    locs, props = find_peaks(signal, height=ph, prominence=pp)
    locs = locs + window_offset
    return locs, props


def extract_single_pulses(signal, locs, pw, rise_offset, args=None):
    pulses = []
    nr_peaks = len(locs)
    len_signal = len(signal)
    singles = np.zeros(nr_peaks, dtype=bool)
    if args is None:
        args = range(nr_peaks)
    for arg in args:
        loc = locs[arg]
        single = 1
        if  arg < nr_peaks - 1 and arg > 0: 
            prev_loc = locs[arg-1]
            next_loc = locs[arg+1]
            if (loc + pw >= next_loc or loc - pw - rise_offset <= prev_loc or loc + pw >= len_signal or loc - rise_offset < 0):
                single = 0
        elif arg == 0 and not arg == nr_peaks - 1:
            next_loc = locs[arg+1]
            if nr_peaks > 1:
                if (loc + pw >= next_loc or loc + pw >= len_signal or loc - rise_offset < 0):
                    single = 0
            else:
                if (loc + pw >= len_signal or loc - rise_offset < 0):
                    single = 0
        elif arg == nr_peaks - 1: 
            prev_loc = locs[arg-1]
            if (loc - pw - rise_offset <= prev_loc or loc + pw >= len_signal or loc - rise_offset < 0):
                single = 0 
        if single:                 
            singles[arg] = 1
            pulse = signal[loc-rise_offset:loc+pw]
            pulses.append(pulse)
    if np.sum(singles):
        pulses_aligned = np.array(pulses).reshape((-1, pw+rise_offset)) 
        return pulses_aligned, singles
    else:
        return  np.empty((1, pw+rise_offset)), singles


# Parameters
file_path = r"Z:\KIDonSun\experiments\Entropy ADR\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800 long\TD_Power\KID3_117dBm__TDvis0_TmK100.bin"  # file path to .bin file
pulse_threshold = 5  # detection threshold in multiples of std
pulse_window = 1000 # pulse window in nr sumples
pulse_offset = 100  # offset to rising edge of extracted pulses
pulse_lifetime = 250  # filter lifetime in nr samples
window_type = 'exp'  # filter type: 'exp', 'box' or 'None'

# Load data
amp, phase = bin2mat(file_path)

# Generate filtering window
exp_filter = get_window(window_type, tau=pulse_lifetime)  # generate exponential filter

# Filter data by convolving with window
filtered_phase = fftconvolve(phase, exp_filter, mode='valid')

# Find the standard deviation of the filtered noise
neg_signal = filtered_phase[filtered_phase<=0]
std = np.std(np.hstack((neg_signal, np.absolute(neg_signal))))

# Set peak finding parameters
height = pulse_threshold*std
prominence = height

# Find peaks
locs, props = find_pks(phase, height, prominence, exp_filter)

# Extract the single pulses from raw data
pulses, single_idx = extract_single_pulses(phase, locs, pulse_window, pulse_offset)
avg_pulse = np.mean(pulses, axis=0)

fig, axes = plt.subplot_mosaic('abc', figsize=(9,3), constrained_layout=True)
time = np.arange(len(phase))
time_filtered = np.arange(len(filtered_phase))
ax = axes ['a']
ax.plot(exp_filter)
ax.set_title('Filter window')
ax = axes['b']
ax.plot(time, phase, lw=.5, label='phase')
ax.plot(time_filtered, filtered_phase, lw=.5, label='filtered phase')
ax.scatter(time[locs], phase[locs], c='r', label='peaks')
ax.axhline(height, ls='--', c='r', lw=1, label='threshold')
ax.legend()
ax.set_title('Timestream')
ax = axes['c']
ax.plot(avg_pulse)
ax.set_title('Average pulse')
plt.show()