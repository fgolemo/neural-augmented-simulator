import os
import time

import h5py

from nas.data import DATA_PATH

#
# # ====== LOADING TIMES
#
# # ====== compressed
#
# f = h5py.File(os.path.join(DATA_PATH, "real-recordings-01-02-10-v1-comp.hdf5"), 'r')
# print (list(f.keys()))
# print (list(f["01"].keys()))
#
#
# start = time.time()
#
# real_posvel = f["01/real_posvel"]
# print (real_posvel.shape)
# for i in range(len(real_posvel)):
#     x = real_posvel[i]
#
# diff = time.time() - start
#
# print (f"compressed: {diff}s")
#
# # ====== uncompressed
#
# f = h5py.File(os.path.join(DATA_PATH, "real-recordings-01-02-10-v1.hdf5"), 'r')
# print (list(f.keys()))
#
#
# start = time.time()
#
# real_posvel = f["01/real_posvel"]
# print (real_posvel.shape)
# for i in range(len(real_posvel)):
#     x = real_posvel[i]
#
# diff = time.time() - start
#
# print (f"uncompressed: {diff}s")


