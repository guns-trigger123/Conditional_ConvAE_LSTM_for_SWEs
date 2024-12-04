import torch
import numpy as np
import os
from torch.utils.data import Dataset


class ShallowWaterReconstructDataset(Dataset):
    def __init__(self, data_path: str, conditions: list):
        self.data_path = data_path
        self.max_timestep = 200
        self.minmax_data = torch.tensor(np.load("./data/minmax/minmax_data.npy"), dtype=torch.float32)
        self.min_vals, self.max_vals = self.minmax_data[0].view(3, 1, 1), self.minmax_data[1].view(3, 1, 1)

        self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i}
                           for (R, Hp) in conditions
                           for i in range(self.max_timestep)]

    def __getitem__(self, item):
        input_dict = self.input_dcit[item]
        R, Hp = input_dict["R"], input_dict["Hp"]
        input_index = input_dict["input_index"]

        # Loading data once for input
        data = np.load(os.path.join(self.data_path, f"R_{R}_Hp_{Hp}.npy"), allow_pickle=True, mmap_mode='r')
        input_data = data[input_index]

        # Normalize data
        input_data = torch.as_tensor(input_data, dtype=torch.float32)
        input_data = (input_data - self.min_vals) / (self.max_vals - self.min_vals)

        return {"input": input_data, "Hp": Hp, "R": R}

    def __len__(self):
        return len(self.input_dcit)


class ShallowWaterLatentPredictDataset(Dataset):
    def __init__(self, data_path: str, conditions: list, rollout=False):
        self.data_path = data_path
        self.rollout = rollout
        self.max_timestep = 200
        self.input_len, self.target_len = 5, 5
        roll = 2 if self.rollout else 1
        self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i, "target_index": i + roll * self.input_len}
                           for (R, Hp) in conditions
                           for i in range(self.max_timestep - roll * self.input_len - self.target_len + 1)]

    def __getitem__(self, item):
        input_dict = self.input_dcit[item]
        R, Hp = input_dict["R"], input_dict["Hp"]
        input_index, target_index = input_dict["input_index"], input_dict["target_index"]

        # Loading latent data once for both input and target
        latent_data = torch.load(os.path.join(self.data_path, f"R_{R}_Hp_{Hp}_latent.pt"), weights_only=True)

        input_data = latent_data[input_index:input_index + self.input_len]
        target_data = latent_data[target_index:target_index + self.target_len]

        return {
            "input": input_data,
            "target": target_data,
            "Hp": Hp,
            "R": R,
            "input_start_timestep": input_index,
            "target_start_timestep": target_index
        }

    def __len__(self):
        return len(self.input_dcit)


class ShallowWaterPredictDataset(Dataset):
    def __init__(self, data_path: str, conditions: list, rollout=False):
        self.data_path = data_path
        self.rollout = rollout
        self.max_timestep = 200
        self.input_len, self.target_len = 5, 5
        self.minmax_data = torch.tensor(np.load("./data/minmax/minmax_data.npy"), dtype=torch.float32)
        self.min_vals, self.max_vals = self.minmax_data[0].view(3, 1, 1), self.minmax_data[1].view(3, 1, 1)

        roll = 2 if self.rollout else 1
        self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i, "target_index": i + roll * self.input_len}
                           for (R, Hp) in conditions
                           for i in range(self.max_timestep - roll * self.input_len - self.target_len + 1)]

    def __getitem__(self, item):
        input_dict = self.input_dcit[item]
        R, Hp = input_dict["R"], input_dict["Hp"]
        input_index, target_index = input_dict["input_index"], input_dict["target_index"]

        # Loading data once for both input and target (since they are from the same file)
        data = np.load(os.path.join(self.data_path, f"R_{R}_Hp_{Hp}.npy"), allow_pickle=True, mmap_mode='r')

        input_data = data[input_index:input_index + self.input_len]
        target_data = data[target_index:target_index + self.target_len]

        # Normalize data
        input_data = torch.as_tensor(input_data, dtype=torch.float32)
        target_data = torch.as_tensor(target_data, dtype=torch.float32)
        input_data = (input_data - self.min_vals) / (self.max_vals - self.min_vals)
        target_data = (target_data - self.min_vals) / (self.max_vals - self.min_vals)

        return {
            "input": input_data,
            "target": target_data,
            "Hp": Hp,
            "R": R,
            "input_start_timestep": input_index,
            "target_start_timestep": target_index
        }

    def __len__(self):
        return len(self.input_dcit)
