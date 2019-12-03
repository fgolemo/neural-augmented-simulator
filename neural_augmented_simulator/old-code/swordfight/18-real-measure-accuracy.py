import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.autograd import Variable
from tqdm import tqdm

from simple_joints_lstm.lstm_simple_net2_real import LstmSimpleNet2Real

DATASET_PATH = "~/data/sim2real/dataset-real-{}-normalized.npz"  # {} is either "train" or "test"

ds_train = np.load(os.path.expanduser(DATASET_PATH.format("train")))

x_train = ds_train["ds_in"]
y_train = ds_train["ds_diff"]

def double_unsqueeze(data):
    return torch.unsqueeze(torch.unsqueeze(torch.from_numpy(data), dim=0), dim=0)


def double_squeeze(data):
    return torch.squeeze(torch.squeeze(data)).data.numpy()


def data_to_var(data):
    return (Variable(double_unsqueeze(data[:12])),
            Variable(double_unsqueeze(data[12:24])),
            Variable(double_unsqueeze(data[24:])))


def se(A, B):
    return ((A - B) ** 2)

out = []

# ["1_v1_3l_128", "2_v1_3l_256", "3_v1_5l_128", "4_v1_5l_256"]

for model in ["5_l3_n128","6_l5_n128","7_l3_n256","8_l5_n256"]:

    model_file = "../trained_models/lstm_real_v2_exp{}_best".format(model)

    stored_model = torch.load(model_file, map_location='cpu')

    lstm = LstmSimpleNet2Real()
    lstm.load_state_dict(stored_model["state_dict"])
    lstm.eval()

    cumm_err = np.zeros(12, dtype=np.float32)


    old_action = x_train[0, 24:].copy()
    action = 0

    for i in tqdm(range(len(x_train))):
        inf = double_squeeze(lstm.forward(data_to_var(x_train[i])))
        serr = se(inf, y_train[i])
        cumm_err += serr


        if not (old_action == x_train[i, 24:]).all():
            action += 1
            old_action = x_train[i, 24:]

            if action == 3:
                lstm.zero_hidden()
                action = 0

    out.append((model, cumm_err))

for model, err in out:
    print (model, err.mean())


## NORMAL MODELS

# 1_v1_3l_128 24016414.0
# 2_v1_3l_256 33650284.0
# 3_v1_5l_128 26463222.0
# 4_v1_5l_256 20966040.0

## "BEST" MODELS

# 1_v1_3l_128 24016414.0
# 2_v1_3l_256 33650284.0
# 3_v1_5l_128 28269048.0
# 4_v1_5l_256 35018588.0


