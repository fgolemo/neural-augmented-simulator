import os

import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import RECORDINGS_PATH

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



# ============= DEBUGGING variant 10 dataset issue


from common import RealRecordingsV1

f = h5py.File(os.path.join(RECORDINGS_PATH, "real-recordings-01-02-10-v1.hdf5"), 'r')

print (f.get("/02/real_posvel").shape)
print (f.get("/10/real_posvel").shape)
print (f.get("/10/actions").shape)

ds1 = RealRecordingsV1("02")
print (len(ds1))

ds2 = RealRecordingsV1("10")
print (len(ds2))

dataloader1 = DataLoader(ds1, batch_size=1, shuffle=True, num_workers=1)

i = 0
for element in enumerate(tqdm(dataloader1, desc="EPISD: ")):
    i+=1

print (f"ds1: {i}")


dataloader2 = DataLoader(ds2, batch_size=1, shuffle=True, num_workers=1)

i = 0
for element in enumerate(tqdm(dataloader2, desc="EPISD: ")):
    i+=1

print (f"ds2: {i}")

