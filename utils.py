import re
import os
import torch
import numpy as np
from shallow_water_dataset import ShallowWaterReconstructDataset, ShallowWaterLatentPredictDataset, ShallowWaterPredictDataset
from recon_model import ConvAutoencoder, MLP
from pred_model import LSTMPredictor, ConditionalLSTMPredictor


def init_recon_model():
    return ConvAutoencoder(), MLP()


def init_pred_model(isConditional=False):
    if isConditional:
        return ConditionalLSTMPredictor()
    else:
        return LSTMPredictor()


def init_recon_data(config: dict, tag: str):
    data_path = f"./data/{tag}/raw"
    num_workers = config["dataset_params"][f"{tag}_num_workers"]
    batch_size = config["dataset_params"][f"{tag}_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    recon_dataset = ShallowWaterReconstructDataset(data_path, conditions)
    recon_dataloader = torch.utils.data.DataLoader(recon_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=num_workers,
                                                   )
    return recon_dataset, recon_dataloader


def init_latent_pred_data(config: dict, tag: str, conv_ae_name: str, shuffle=True, rollout=False):
    data_path = f"./data/{tag}/latent128/{conv_ae_name}"
    num_workers = config["dataset_params"][tag + "_num_workers"]
    batch_size = config["dataset_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.pt$", i)]
    if shuffle == False:
        conditions.sort()
    latent_pred_dataset = ShallowWaterLatentPredictDataset(data_path, conditions, rollout)
    latent_pred_dataloader = torch.utils.data.DataLoader(latent_pred_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=shuffle,
                                                         num_workers=num_workers,
                                                         )
    return latent_pred_dataset, latent_pred_dataloader


def init_pred_data(config: dict, tag: str, rollout=False):
    data_path = f"./data/{tag}/raw"
    num_workers = config["dataset_params"][tag + "_num_workers"]
    batch_size = config["dataset_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    conditions.sort()
    pred_dataset = ShallowWaterPredictDataset(data_path, conditions, rollout)
    pred_dataloader = torch.utils.data.DataLoader(pred_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_workers,
                                                  )
    return pred_dataset, pred_dataloader


def load_recon_checkpoint(SAVED_DIRECTORY, SAVED_PREFIX, conv_ae, opt_conv_ae, sch_conv_ae, mlp, opt_mlp, sch_mlp):
    checkpoint_before = torch.load(SAVED_DIRECTORY + "checkpoint_" + SAVED_PREFIX + ".pt")
    conv_ae.load_state_dict(checkpoint_before['conv_ae'])
    opt_conv_ae.load_state_dict(checkpoint_before['opt_conv_ae'])
    sch_conv_ae.load_state_dict(checkpoint_before['sch_conv_ae'])
    mlp.load_state_dict(checkpoint_before['mlp'])
    opt_mlp.load_state_dict(checkpoint_before['opt_mlp'])
    sch_mlp.load_state_dict(checkpoint_before['sch_mlp'])
    last_epoch = checkpoint_before['epoch']
    return conv_ae, opt_conv_ae, sch_conv_ae, mlp, opt_mlp, sch_mlp, last_epoch


def load_recon_best_epoch(SAVED_DIRECTORY, SAVED_PREFIX, conv_ae, device, val_dataloader):
    best_epoch = torch.load(SAVED_DIRECTORY + SAVED_PREFIX + "_best_epoch.pt")
    conv_ae.load_state_dict(torch.load(SAVED_DIRECTORY + SAVED_PREFIX + "_best.pt"))
    uvh_mean_err_list = []
    for iter, batch in enumerate(val_dataloader):
        conv_ae.eval()
        batch_input = batch["input"].to(device)
        results = conv_ae(batch_input)
        batch_err = torch.abs(batch_input - results)
        rela_batch_err = batch_err / (1 + batch_input)

        uvh_mean_err = torch.mean(rela_batch_err, dim=(0, 2, 3))
        uvh_mean_err = uvh_mean_err.cpu().detach().numpy()
        uvh_mean_err_list.append(uvh_mean_err)

    uvh_list_mean = sum(uvh_mean_err_list) / len(val_dataloader)

    return best_epoch, uvh_list_mean


def load_pred_checkpoint(SAVED_DIRECTORY, SAVED_PREFIX, lstm, opt, sch):
    checkpoint_before = torch.load(SAVED_DIRECTORY + "checkpoint_" + SAVED_PREFIX + ".pt")
    lstm.load_state_dict(checkpoint_before['lstm'])
    opt.load_state_dict(checkpoint_before['opt'])
    sch.load_state_dict(checkpoint_before['sch'])
    last_epoch = checkpoint_before['epoch']
    return lstm, opt, sch, last_epoch


def load_pred_best_epoch(SAVED_DIRECTORY, SAVED_PREFIX, lstm, device, val_dataloader, isConditional=False):
    best_epoch = torch.load(SAVED_DIRECTORY + SAVED_PREFIX + "_best_epoch.pt")
    lstm.load_state_dict(torch.load(SAVED_DIRECTORY + SAVED_PREFIX + "_best.pt"))
    pred_mean_err_list = []
    for iter, batch in enumerate(val_dataloader):
        lstm.eval()
        batch_input, batch_target = batch["input"].to(device), batch["target"].to(device)
        if isConditional:
            R, Hp = batch["R"].reshape(-1, 1, 1) / 40.0, batch["Hp"].reshape(-1, 1, 1) / 20.0
            R, Hp = R.repeat(1, 5, 1), Hp.repeat(1, 5, 1)
            reg_labels = torch.cat([R, Hp], dim=2).to(device)
            batch_input = torch.cat([batch_input, reg_labels], dim=2)
        batch_pred = lstm(batch_input)
        batch_err = torch.abs(batch_target - batch_pred)

        batch_mean_err = torch.mean(batch_err, dim=(0, 2))
        batch_mean_err = batch_mean_err.cpu().detach().numpy()
        pred_mean_err_list.append(batch_mean_err)

    pred_list_mean = sum(pred_mean_err_list) / len(val_dataloader)

    return best_epoch, pred_list_mean
