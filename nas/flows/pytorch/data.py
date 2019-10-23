from functools import partial
import numpy as np

from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset

# --------------------
# Helper functions
# --------------------


def logit(x, eps=1e-5):
    x.clamp_(eps, 1 - eps)
    return x.log() - (1 - x).log()


def one_hot(x, label_size):
    out = torch.zeros(len(x), label_size).to(x.device)
    out[torch.arange(len(x)), x] = 1
    return out


def load_dataset(name):
    exec('from datasets.{} import {}'.format(name.lower(), name))
    return locals()[name]


# --------------------
# Dataloaders
# --------------------

def fetch_dataloaders(dataset_name, batch_size, device, flip_toy_var_order=False, toy_train_size=25000, toy_test_size=5000):

    # grab datasets
    # use the constructors by MAF authors
    if dataset_name in ['GAS', 'POWER', 'HEPMASS', 'MINIBOONE', 'BSDS300']:
        dataset = load_dataset(dataset_name)()

        # join train and val data again
        train_data = np.concatenate((dataset.trn.x, dataset.val.x), axis=0)

        # construct datasets
        train_dataset = TensorDataset(
            torch.from_numpy(train_data.astype(np.float32)))
        test_dataset = TensorDataset(
            torch.from_numpy(dataset.tst.x.astype(np.float32)))

        input_dims = dataset.n_dims
        label_size = None
        lam = None

    elif dataset_name in ['MNIST']:
        dataset = load_dataset(dataset_name)()

        # join train and val data again
        train_x = np.concatenate(
            (dataset.trn.x, dataset.val.x), axis=0).astype(np.float32)
        train_y = np.concatenate(
            (dataset.trn.y, dataset.val.y), axis=0).astype(np.float32)

        # construct datasets
        train_dataset = TensorDataset(
            torch.from_numpy(train_x), torch.from_numpy(train_y))
        test_dataset = TensorDataset(torch.from_numpy(dataset.tst.x.astype(np.float32)),
                                     torch.from_numpy(dataset.tst.y.astype(np.float32)))

        input_dims = dataset.n_dims
        label_size = 10
        lam = dataset.alpha

    elif dataset_name in ['TOY', 'MOONS']:  # use own constructors

        n_samples = 1000
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
        train_x, train_y = noisy_moons
        train_x = StandardScaler().fit_transform(train_x)

        # pdb.set_trace()
        train_dataset = TensorDataset(
            torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32)))
        test_dataset = TensorDataset(
            torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32)))

        train_dataset.base_dist = torch.distributions.MultivariateNormal(
            torch.zeros(2), scale_tril=torch.diag(torch.ones(2)))

        input_dims = train_x.shape[1]
        label_size = 2
        lam = None

    else:
        raise ValueError('Unrecognized dataset.')

    # keep input dims, input size and label size
    train_dataset.input_dims = input_dims
    train_dataset.input_size = int(np.prod(input_dims))
    train_dataset.label_size = label_size
    train_dataset.lam = lam

    test_dataset.input_dims = input_dims
    test_dataset.input_size = int(np.prod(input_dims))
    test_dataset.label_size = label_size
    test_dataset.lam = lam

    # construct dataloaders
    kwargs = {'num_workers': 1,
              'pin_memory': True} if device.type is 'cuda' else {}

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader
