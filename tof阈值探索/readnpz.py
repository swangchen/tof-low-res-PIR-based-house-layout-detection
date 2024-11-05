import numpy as np

with np.load("20241016_145743_tof_rawdataL2.npz",allow_pickle=True) as data:
    for arr_name in data:
        print(f"Contents of {arr_name}:\n", data[arr_name])