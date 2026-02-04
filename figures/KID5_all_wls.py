import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import pulse_analysis, find_threshold
import numpy as np
import matplotlib.pyplot as plt


kid = 5
pread = 113
file_type = 'vis'
filter = 'exp'
pw = 1500
pw_offset = 100
tqp = [200, 600]
mph = np.array([5, 30])
mpp = mph[0]
iterate = 0
stds = [1, 2, 3, 4, 5, 6]

# dir = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800 long\TD_Power"
# chuncksize = 10
# nr_chuncks = 1
# lifetime = 250

# dir = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs LN2 load - long\TD_Power"
# chuncksize = 10
# nr_chuncks = 1
# lifetime = 250

# dir = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
# chuncksize = 40
# nr_chuncks = 1
# lifetime = 250
# iterate = 0

# dir = r'D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB24K\TD_Power'
# kid = 5
# pread = 109
# chuncksize = 40
# nr_chuncks = 1
# mph = np.array([4, 30])
# mpp = mph[0]
# lifetime = 250
# iterate = 0

# find_threshold(dir, kid, pread, file_type, chuncksize, pw, pw_offset, filter, lifetime, stds)
pulse_analysis(dir, kid, pread, file_type, chuncksize, nr_chuncks, pw, pw_offset, filter, lifetime, mph, mpp, iterate=iterate, plot=True, coord='circle', exclude_dc=True)
plt.show()