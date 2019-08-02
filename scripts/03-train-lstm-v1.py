import os

from comet_ml import Experiment

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from nas.data import MODELS_PATH
from nas.data.datasets import RealRecordingsV1
from nas.models.networks import LstmNetRealv1
from nas.utils import log_parameters

experiment = Experiment(
    api_key="ZfKpzyaedH6ajYSiKmvaSwyCs",
    project_name="nas-v2",
    workspace="fgolemo")

HIDDEN_NODES = 128
LSTM_LAYERS = 3
EPOCHS = 5
VARIANT = "01"
EXPERIMENT_ID = 1  # increment and then commit to github when you change training/network code
MODEL_PATH = os.path.join(
    MODELS_PATH, f"model-"
    f"exp{EXPERIMENT_ID}-"
    f"h{HIDDEN_NODES}-"
    f"l{LSTM_LAYERS}-"
    f"v{VARIANT}-"
    f"e{EPOCHS}.pth")

log_parameters(
    experiment,
    hidden_nodes=HIDDEN_NODES,
    lstm_layers=LSTM_LAYERS,
    epochs=EPOCHS,
    variant=VARIANT,
    experiment_id=EXPERIMENT_ID)

dataset_train = RealRecordingsV1(dtype=VARIANT)

# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader_train = DataLoader(
    dataset_train, batch_size=1, shuffle=True, num_workers=1)

net = LstmNetRealv1(
    n_input_state_sim=12,
    n_input_state_real=12,
    n_input_actions=6,
    nodes=HIDDEN_NODES,
    layers=LSTM_LAYERS)

if torch.cuda.is_available():
    net = net.cuda()

net = net.float()


def extract(dataslice):
    x, y = (dataslice["x"].transpose(0, 1).float(),
            dataslice["y"].transpose(0, 1).float())

    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    return x, y


def save(model):
    torch.save(model.state_dict(), MODEL_PATH)


loss_function = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

for epoch in trange(EPOCHS, desc="EPOCH: "):

    loss_epoch = 0
    diff_epoch = 0

    for epi_idx, epi_data in enumerate(tqdm(dataloader_train, desc="EPISD: ")):
        x, y = extract(epi_data)

        net.zero_grad()
        net.zero_hidden()
        optimizer.zero_grad()

        delta = net.forward(x)

        # for idx in range(len(x)):
        #     print(idx, "=")
        #     print("real t1_x:", np.around(x[idx, 0, 12:24].cpu().data.numpy(), 2))
        #     print("sim_ t2_x:", np.around(x[idx, 0, :12].cpu().data.numpy(), 2))
        #     print("action__x:", np.around(x[idx, 0, 24:].cpu().data.numpy(), 2))
        #     print("real t2_x:",
        #           np.around(x[idx, 0, :12].cpu().data.numpy() + y[idx, 0].cpu().data.numpy(), 2))
        #     print("real t2_y:",
        #           np.around(x[idx, 0, :12].cpu().data.numpy() + delta[idx, 0].cpu().data.numpy(), 2))
        #     print("delta___x:",
        #           np.around(y[idx, 0].cpu().data.numpy(), 3))
        #     print("delta___y:",
        #           np.around(delta[idx, 0].cpu().data.numpy(), 3))
        #     print("===")

        loss = loss_function(delta, y)
        loss.backward()
        optimizer.step()

        loss_episode = loss.clone().cpu().data.numpy()
        diff_episode = (y.cpu().data.numpy()**2).mean(axis=None)

        experiment.log_metric("loss episode", loss_episode)
        experiment.log_metric("diff episode", diff_episode)

        loss.detach_()
        net.hidden[0].detach_()
        net.hidden[1].detach_()

        loss_epoch += loss_episode
        diff_epoch += diff_episode

    print(loss_epoch, diff_epoch)
    experiment.log_metric("loss epoch", loss_epoch / len(dataloader_train))
    experiment.log_metric("diff epoch", diff_epoch / len(dataloader_train))

    save(net)

    #TODO take a single trajectory and roll it out based on the real data and based on the LSTM corrections, then print it to matplotlib and upload to comet via
    # experiment.log_figure()

    #TODO split into train and test set
