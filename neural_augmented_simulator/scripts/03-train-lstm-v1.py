import os

from comet_ml import Experiment

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import argparse
from neural_augmented_simulator.common import MODELS_PATH
from neural_augmented_simulator.common.nas.data.datasets import RealRecordingsV1
from neural_augmented_simulator.common.nas.models.networks import LstmNetRealv1
from neural_augmented_simulator.common.nas.utils import log_parameters
from torch.utils.data.sampler import SubsetRandomSampler
experiment = Experiment(
    api_key="ZfKpzyaedH6ajYSiKmvaSwyCs",
    project_name="nas-v2",
    workspace="fgolemo")
# experiment.add_tag("debugging-10")

parser = argparse.ArgumentParser(description='LSTM train')
parser.add_argument('--variant', default='10', help='Variant')
parser.add_argument('--approach', default='goal-babbling', help='Approach')
parser.add_argument('--hidden_nodes', default=128, help='Number of Hidden Nodes in LSTM')
parser.add_argument('--lstm_layers', default=3, help='Number of LSTM Layers')
parser.add_argument('--experiment_id', default=1, help='Experiment ID')
parser.add_argument('--noise-type', default='action-noise', help='Experiment ID')
parser.add_argument('--epochs', default=5, help='Epochs')
args = parser.parse_args()


HIDDEN_NODES = args.hidden_nodes
LSTM_LAYERS = args.lstm_layers
EPOCHS = args.epochs
VARIANT = args.variant
EXPERIMENT_ID = args.experiment_id  # increment and then commit to github when you change training/network code

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

dataset = RealRecordingsV1(dtype=VARIANT, approach=args.approach, noise_type=args.noise_type)
dataset_size = len(dataset)
indices = list(range(dataset_size))
# SPLITING THE DATASET INTO TRAIN AND TEST
test_split = 0.2
split = int(np.floor(test_split * dataset_size))
train_indices, test_indices = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)


# batch size has to be 1, otherwise the LSTM doesn't know what to do
dataloader_train = DataLoader(
    dataset, batch_size=1, num_workers=1, sampler=train_sampler)
dataloader_test = DataLoader(
    dataset, batch_size=1, num_workers=1, sampler=test_sampler)

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
    test_loss_epoch = 0
    test_diff_epoch = 0

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
            if (epi_idx + 1) % 10 == 0:
                # Testing
                with torch.no_grad():
                    net.eval()
                    for test_epi_idx, test_epi_data in enumerate(tqdm(dataloader_test, desc="EPISD: ")):
                        test_x, test_y = extract(test_epi_data)
                        test_delta = net.forward(test_x)
                        test_loss_episode = loss_function(test_delta, test_y)
                        test_diff_episode = (test_y.cpu().data.numpy() ** 2).mean(axis=None)
                        experiment.log_metric("test loss episode", test_loss_episode.cpu().data.numpy())
                        test_loss_epoch += test_loss_episode
                        test_diff_epoch += test_diff_episode

    experiment.log_metric("loss epoch", loss_epoch / len(dataloader_train))
    experiment.log_metric("diff epoch", diff_epoch / len(dataloader_train))

    save(net)


