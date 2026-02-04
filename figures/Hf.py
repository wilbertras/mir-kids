import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import pulse_analysis, find_threshold
import numpy as np
import matplotlib.pyplot as plt


kid = 5
pread = 109
file_type = 'vis'
filter = 'exp'
pw = 1000
pw_offset = 100
tqp = [250, 500]
mph = np.array([5, 100])
mpp = mph[0]
iterate = 0
stds = [1, 2, 3, 4, 5, 6]

dir = r"Z:\KIDonSun\experiments\Entropy ADR\LT365Chip8_HfSapphire_BF_20240911\KID5 402nm 7uW 24dB\TD_Power"
chuncksize = 10
nr_chuncks = 1
lifetime = 100



# find_threshold(dir, kid, pread, file_type, chuncksize, pw, pw_offset, filter, lifetime, stds)
pulse_analysis(dir, kid, pread, file_type, chuncksize, nr_chuncks, pw, pw_offset, filter, lifetime, mph, mpp, iterate=iterate, plot=True, coord='circle', exclude_dc=True, fit_tqp=tqp)
plt.show()