import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import os
import glob
import h5py
from tqdm import tqdm

# train_noisy = sorted(glob.glob(os.path.join("data", "rec_noisy", "*.npy")))
# train_clean = sorted(glob.glob(os.path.join("data", "rec_clean", "*.npy")))
test_noisy = sorted(glob.glob(os.path.join("data", "test_noisy", "*.npy")))
test_clean = sorted(glob.glob(os.path.join("data", "test_clean", "*.npy")))

# with h5py.File(os.path.join("data", "input.h5"), "w") as f:
#     n = 0
#     for file in tqdm(range(len(train_noisy))):
#         noisy_img = np.load(train_noisy[file])
#         f.create_dataset(str(n), data=noisy_img)
#         n += 1
#
# with h5py.File(os.path.join("data", "target.h5"), "w") as f2:
#     n = 0
#     for file in tqdm(range(len(train_clean))):
#         clean_img = np.load(train_clean[file])
#         f2.create_dataset(str(n), data=clean_img)
#         n += 1

with h5py.File(os.path.join("data", "test_inp.h5"), "w") as f:
    n = 0
    for file in tqdm(range(len(test_noisy))):
        noisy_img = np.load(test_noisy[file])
        f.create_dataset(str(n), data=noisy_img)
        n += 1

with h5py.File(os.path.join("data", "test_target.h5"), "w") as f2:
    n = 0
    for file in tqdm(range(len(test_clean))):
        clean_img = np.load(test_clean[file])
        f2.create_dataset(str(n), data=clean_img)
        n += 1

class CT_Dataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.h5f_t = h5py.File(os.path.join(filename, "target.h5"), "r")
        self.keys_t = list(self.h5f_t.keys())
        self.h5f_n = h5py.File(os.path.join(filename, "input.h5"), "r")
        self.keys_n = list(self.h5f_n.keys())

    def __len__(self):
        return len(self.keys_t)

    def __getitem__(self, index):
        name_t = self.keys_t[index]
        name_n = self.keys_n[index]
        data_t = np.array(self.h5f_t[name_t])
        data_n = np.array(self.h5f_n[name_n])
        data_t = np.expand_dims(data_t, axis=0)
        data_n = np.expand_dims(data_n, axis=0)
        inp = torch.Tensor(data_n)
        tgt = torch.Tensor(data_t)

        return inp, tgt

class test_dataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.h5f_t = h5py.File(os.path.join(filename, "test_target.h5"), "r")
        self.keys_t = list(self.h5f_t.keys())
        self.h5f_n = h5py.File(os.path.join(filename, "test_inp.h5"), "r")
        self.keys_n = list(self.h5f_n.keys())

    def __len__(self):
        return len(self.keys_t)

    def __getitem__(self, index):
        name_t = self.keys_t[index]
        name_n = self.keys_n[index]
        data_t = np.array(self.h5f_t[name_t])
        data_n = np.array(self.h5f_n[name_n])
        data_t = np.expand_dims(data_t, axis=0)
        data_n = np.expand_dims(data_n, axis=0)
        inp = torch.Tensor(data_n)
        tgt = torch.Tensor(data_t)

        return inp, tgt













