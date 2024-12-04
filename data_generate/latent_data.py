import re
import os
import torch
import numpy as np
from recon_model import ConvAutoencoder


def _init_directory(latent_name, conv_ae_name):
    dirs = [
        f"../data/train/{latent_name}/{conv_ae_name}/",
        f"../data/val/{latent_name}/{conv_ae_name}/",
        f"../data/test/{latent_name}/{conv_ae_name}/"
    ]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def _load_minmax():
    minmax_data = np.load("../data/minmax/minmax_data.npy")
    minmax_data = torch.tensor(minmax_data, dtype=torch.float32)
    min_vals, max_vals = minmax_data[0].view(1, 3, 1, 1), minmax_data[1].view(1, 3, 1, 1)
    return min_vals, max_vals


def gen_latent_data(conv_ae_type, model_number):
    conv_ae_name = f"{conv_ae_type}_{model_number}"
    _init_directory("latent128", conv_ae_name)
    min_vals, max_vals = _load_minmax()
    conv_ae = ConvAutoencoder()
    conv_ae.load_state_dict(torch.load(f"../saved_models/latent128/{conv_ae_name}/{conv_ae_type}_best.pt", map_location='cpu'))

    for tag in ["train", "val", "test"]:
        data_path = f"../data/{tag}/raw"
        conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                      if re.search(r"\.npy$", i)]
        for (R, Hp) in conditions:
            input_data = np.load(f"{data_path}/R_{R}_Hp_{Hp}.npy", allow_pickle=True, mmap_mode='r')
            input_data = torch.as_tensor(input_data.copy(), dtype=torch.float32)
            input_data = (input_data - min_vals) / (max_vals - min_vals)
            latent_data = conv_ae.encoder(input_data).detach()
            torch.save(latent_data, f"../data/{tag}/latent128/{conv_ae_name}/R_{R}_Hp_{Hp}_latent.pt")


if __name__ == '__main__':
    conv_ae_type, model_number = "ae", "1"
    gen_latent_data(conv_ae_type, model_number)
