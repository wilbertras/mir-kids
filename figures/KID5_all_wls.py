import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import pulse_analysis
import numpy as np
import matplotlib.pyplot as plt


kid = 5
pread = 113
file_type = 'vis'
pw = 1500
pw_offset = 100
filter = 'exp'
lifetime = 250
tqp = [200, 600]
iterate = 0


dir = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800 long\TD_Power"
chuncksize = 10
nr_chuncks = None
mph = np.array([5, 40])
mpp = mph[0]

# dir = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs LN2 load - long\TD_Power"
# chuncksize = 10
# nr_chuncks = 1
# mph = np.array([5, 20])
# mpp = mph[0]

# dir = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
# dir = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\KID5_all_temps\100K"
# chuncksize = 10
# nr_chuncks = 1
# mph = np.array([5, 20])
# mpp = mph[0]

pulse_analysis(dir, kid, pread, file_type, chuncksize, nr_chuncks, pw, pw_offset, filter, lifetime, mph, mpp, iterate=iterate, plot=True, coord='circle', exclude_dc=True, fit_tqp=tqp)
plt.show()