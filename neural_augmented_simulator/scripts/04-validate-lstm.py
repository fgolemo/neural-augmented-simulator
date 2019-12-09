import os

from comet_ml import Experiment

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import argparse
from nas.data import MODELS_PATH
from nas.data.datasets import RealRecordingsV1
from nas.models.networks import LstmNetRealv1
from nas.utils import log_parameters
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


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




net = LstmNetRealv1(
    n_input_state_sim=12,
    n_input_state_real=12,
    n_input_actions=6,
    nodes=HIDDEN_NODES,
    layers=LSTM_LAYERS)

if torch.cuda.is_available():
    net = net.cuda()



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


seed = [100, 200, 300]
approach = ['motor-babbling', 'goal-babbling']
variant = ['01', '02', '10']
evaluations = []
for app in approach:
    for var in variant:
        dataset = RealRecordingsV1(dtype=var, approach=app, noise_type=args.noise_type)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        # SPLITING THE DATASET INTO TRAIN AND TEST
        test_split = 0.2
        split = int(np.floor(test_split * dataset_size))
        train_indices, test_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        dataloader_test = DataLoader(
            dataset, batch_size=1, num_workers=1, sampler=test_sampler)

        MODEL_PATH = os.path.join(
            MODELS_PATH, f"model-"
                         f"exp{EXPERIMENT_ID}-"
                         f"h{HIDDEN_NODES}-"
                         f"l{LSTM_LAYERS}-"
                         f"v{var}-"
                         f"{app}-"
                         f"e{EPOCHS}.pth")
        net.load_state_dict(torch.load(MODEL_PATH))
        print(f'Model {MODEL_PATH} is loaded')
        net = net.float()
        final_eval = []
        for s in seed:
            torch.manual_seed(s)
            final_loss_episode = 0
            with torch.no_grad():
                net.eval()
                for test_epi_idx, test_epi_data in enumerate(tqdm(dataloader_test, desc="EPISD: ")):
                    test_x, test_y = extract(test_epi_data)
                    test_delta = net.forward(test_x)
                    final_loss_episode += loss_function(test_delta, test_y)
                    net.hidden[0].detach_()
                    net.hidden[1].detach_()
            final_eval.append(final_loss_episode.cpu().data.numpy()/len(dataloader_test))
        evaluations.append(final_eval)
labels = ['MB-01', 'MB-02', 'MB-10', 'GB-01', 'GB-02', 'GB-10']
_, ax1 = plt.subplots()
ax1.boxplot(evaluations)
ax1.set_ylabel('Average Loss')
ax1.set_title('Final LSTM losses of all approaches')
plt.xticks(range(1, len(labels) + 1), labels)
figure_name = 'box_and_whiskers_for_lstm.png'
plt.savefig(figure_name)


