import torch
import yaml
import numpy as np
from utils import init_recon_model, init_pred_model, init_latent_pred_data, init_pred_data
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torcheval.metrics.functional import peak_signal_noise_ratio


def mae(conv_ae, lstm, recon_dataloader, latent_dataloader, isConditional, rollout=False):
    # turn to eval mode
    lstm.eval()
    conv_ae.eval()
    # init recon & latent error list
    uvh_mean_err_list, latent_mean_err_list = [], []
    for iter, (batch_latent, batch_recon) in enumerate(zip(latent_dataloader, recon_dataloader)):
        # latent input & labels
        batch_latent_input, batch_latent_labels = batch_latent["input"].to(device), batch_latent["target"].to(device)
        # recon labels
        batch_recon_labels = batch_recon["target"].to(device)
        # pred latent forward
        if rollout == False:
            if isConditional:
                R_0, Hp_0 = batch_latent["R"].reshape(-1, 1, 1) / 40.0, batch_latent["Hp"].reshape(-1, 1, 1) / 20.0
                R, Hp = R_0.repeat(1, 5, 1), Hp_0.repeat(1, 5, 1)
                batch_concate = torch.cat([R, Hp], dim=2).to(device)
                batch_sup_weight = Hp_0.squeeze(2).to(device)
                batch_latent_input = torch.cat([batch_latent_input, batch_concate], dim=2)
                batch_latent_pred = lstm(batch_latent_input, batch_sup_weight).detach().squeeze()
            else:
                batch_latent_pred = lstm(batch_latent_input).detach().squeeze()
        else:
            if isConditional:
                R_0, Hp_0 = batch_latent["R"].reshape(-1, 1, 1) / 40.0, batch_latent["Hp"].reshape(-1, 1, 1) / 20.0
                R, Hp = R_0.repeat(1, 5, 1), Hp_0.repeat(1, 5, 1)
                batch_concate = torch.cat([R, Hp], dim=2).to(device)
                batch_sup_weight = Hp_0.squeeze(2).to(device)
                batch_latent_input = torch.cat([batch_latent_input, batch_concate], dim=2)
                batch_latent_pred = lstm(batch_latent_input, batch_sup_weight).detach()
                batch_latent_input_sec = torch.cat([batch_latent_pred, batch_concate], dim=2)
                batch_latent_pred_sec = lstm(batch_latent_input_sec, batch_sup_weight).detach().squeeze()
                batch_latent_pred = batch_latent_pred_sec
            else:
                batch_latent_pred = lstm(batch_latent_input).detach()
                batch_latent_pred_sec = lstm(batch_latent_pred).detach().squeeze()
                batch_latent_pred = batch_latent_pred_sec
        # pred params
        batch_size = batch_latent_input.shape[0]
        target_len = batch_latent_labels.shape[1]
        latent_dim = batch_latent_labels.shape[2]
        # recon forward
        batch_recon_pred = conv_ae.decoder(batch_latent_pred.reshape(batch_size * target_len, latent_dim)).detach()
        batch_recon_pred = batch_recon_pred.reshape(batch_size, target_len, 3, 128, 128)
        # compute recon relative error
        batch_recon_pred = batch_recon_pred.detach().cpu().numpy()
        batch_recon_labels = batch_recon_labels.cpu().numpy()
        recon_err = np.abs(batch_recon_pred - batch_recon_labels)
        recon_rela_err = recon_err / (1 + batch_recon_labels)
        uvh_mean_err = np.mean(recon_rela_err, axis=(0, 1, 3, 4))
        uvh_mean_err_list.append(uvh_mean_err)
        if rollout == False:
            print(f"relative uvh mean error: {uvh_mean_err}")
        else:
            print(f"rollout relative uvh mean error: {uvh_mean_err}")
        # compute latent error
        batch_latent_labels = batch_latent_labels.detach().cpu().numpy()
        batch_latent_pred = batch_latent_pred.detach().cpu().numpy()
        latent_err = np.abs(batch_latent_pred - batch_latent_labels)
        latent_mean_err = np.mean(latent_err, axis=(0, 1, 2))
        latent_mean_err_list.append(latent_mean_err)
        if rollout == False:
            print(f"latent mean error: {latent_mean_err}")
        else:
            print(f"rollout latent mean error: {latent_mean_err}")
    if rollout == False:
        print(f"uvh_mean_err: {sum(uvh_mean_err_list) / len(recon_dataloader)}")
        print(f"latent_mean_err: {sum(latent_mean_err_list) / len(latent_dataloader)}")
    else:
        print(f"rollout uvh_mean_err: {sum(uvh_mean_err_list) / len(recon_dataloader)}")
        print(f"rollout latent_mean_err: {sum(latent_mean_err_list) / len(latent_dataloader)}")


def ssim(conv_ae, lstm, recon_dataloader, latent_dataloader, isConditional, rollout=False):
    # turn to eval mode
    lstm.eval()
    conv_ae.eval()
    # init ssim_calculator & ssim list
    ssim_calculator = StructuralSimilarityIndexMeasure(data_range=1.0)
    ssim_values = []
    for iter, (batch_latent, batch_recon) in enumerate(zip(latent_dataloader, recon_dataloader)):
        # latent input & labels
        batch_latent_input, batch_latent_labels = batch_latent["input"].to(device), batch_latent["target"].to(device)
        # recon labels
        batch_recon_labels = batch_recon["target"].to(device)
        # pred latent forward
        if rollout == False:
            if isConditional:
                R_0, Hp_0 = batch_latent["R"].reshape(-1, 1, 1) / 40.0, batch_latent["Hp"].reshape(-1, 1, 1) / 20.0
                R, Hp = R_0.repeat(1, 5, 1), Hp_0.repeat(1, 5, 1)
                batch_concate = torch.cat([R, Hp], dim=2).to(device)
                batch_sup_weight = Hp_0.squeeze(2).to(device)
                batch_latent_input = torch.cat([batch_latent_input, batch_concate], dim=2)
                batch_latent_pred = lstm(batch_latent_input, batch_sup_weight).detach().squeeze()
            else:
                batch_latent_pred = lstm(batch_latent_input).detach().squeeze()
        else:
            if isConditional:
                R_0, Hp_0 = batch_latent["R"].reshape(-1, 1, 1) / 40.0, batch_latent["Hp"].reshape(-1, 1, 1) / 20.0
                R, Hp = R_0.repeat(1, 5, 1), Hp_0.repeat(1, 5, 1)
                batch_concate = torch.cat([R, Hp], dim=2).to(device)
                batch_sup_weight = Hp_0.squeeze(2).to(device)
                batch_latent_input = torch.cat([batch_latent_input, batch_concate], dim=2)
                batch_latent_pred = lstm(batch_latent_input, batch_sup_weight).detach()
                batch_latent_input_sec = torch.cat([batch_latent_pred, batch_concate], dim=2)
                batch_latent_pred_sec = lstm(batch_latent_input_sec, batch_sup_weight).detach().squeeze()
                batch_latent_pred = batch_latent_pred_sec
            else:
                batch_latent_pred = lstm(batch_latent_input).detach()
                batch_latent_pred_sec = lstm(batch_latent_pred).detach().squeeze()
                batch_latent_pred = batch_latent_pred_sec
        # pred params
        batch_size = batch_latent_input.shape[0]
        target_len = batch_latent_labels.shape[1]
        latent_dim = batch_latent_labels.shape[2]
        # recon forward
        batch_recon_pred = conv_ae.decoder(batch_latent_pred.reshape(batch_size * target_len, latent_dim)).detach()
        batch_recon_pred = batch_recon_pred.reshape(batch_size, target_len, 3, 128, 128)
        # compute recon ssim
        batch_recon_pred = batch_recon_pred.detach().cpu()
        batch_recon_labels = batch_recon_labels.cpu()
        print(f"batch_recon_pred shape: {batch_recon_pred.shape}")
        ssim_value = ssim_calculator(batch_recon_pred, batch_recon_labels)
        ssim_values.append(ssim_value.unsqueeze(0))
        if rollout == False:
            print(f"batch_recon_pred $ batch_recon_labels ssim: {ssim_value}")
        else:
            print(f"rollout batch_recon_pred $ batch_recon_labels ssim: {ssim_value}")
    if rollout == False:
        print(f"Total ssim values: {ssim_values}")
        ssim_values_tensor = torch.cat(ssim_values)
        ssim_values_mean = torch.mean(ssim_values_tensor)
        print(f"Total ssim mean: {ssim_values_mean}")
    else:
        print(f"rollout Total ssim values: {ssim_values}")
        ssim_values_tensor = torch.cat(ssim_values)
        ssim_values_mean = torch.mean(ssim_values_tensor)
        print(f"rollout Total ssim mean: {ssim_values_mean}")


def psnr(conv_ae, lstm, recon_dataloader, latent_dataloader, isConditional, rollout=False):
    # turn to eval mode
    lstm.eval()
    conv_ae.eval()
    # init psnr list
    psnr_values = []
    for iter, (batch_latent, batch_recon) in enumerate(zip(latent_dataloader, recon_dataloader)):
        # latent input & labels
        batch_latent_input, batch_latent_labels = batch_latent["input"].to(device), batch_latent["target"].to(device)
        # recon labels
        batch_recon_labels = batch_recon["target"].to(device)
        # pred latent forward
        if rollout == False:
            if isConditional:
                R_0, Hp_0 = batch_latent["R"].reshape(-1, 1, 1) / 40.0, batch_latent["Hp"].reshape(-1, 1, 1) / 20.0
                R, Hp = R_0.repeat(1, 5, 1), Hp_0.repeat(1, 5, 1)
                batch_concate = torch.cat([R, Hp], dim=2).to(device)
                batch_sup_weight = Hp_0.squeeze(2).to(device)
                batch_latent_input = torch.cat([batch_latent_input, batch_concate], dim=2)
                batch_latent_pred = lstm(batch_latent_input, batch_sup_weight).detach().squeeze()
            else:
                batch_latent_pred = lstm(batch_latent_input).detach().squeeze()
        else:
            if isConditional:
                R_0, Hp_0 = batch_latent["R"].reshape(-1, 1, 1) / 40.0, batch_latent["Hp"].reshape(-1, 1, 1) / 20.0
                R, Hp = R_0.repeat(1, 5, 1), Hp_0.repeat(1, 5, 1)
                batch_concate = torch.cat([R, Hp], dim=2).to(device)
                batch_sup_weight = Hp_0.squeeze(2).to(device)
                batch_latent_input = torch.cat([batch_latent_input, batch_concate], dim=2)
                batch_latent_pred = lstm(batch_latent_input, batch_sup_weight).detach()
                batch_latent_input_sec = torch.cat([batch_latent_pred, batch_concate], dim=2)
                batch_latent_pred_sec = lstm(batch_latent_input_sec, batch_sup_weight).detach().squeeze()
                batch_latent_pred = batch_latent_pred_sec
            else:
                batch_latent_pred = lstm(batch_latent_input).detach()
                batch_latent_pred_sec = lstm(batch_latent_pred).detach().squeeze()
                batch_latent_pred = batch_latent_pred_sec
        # pred params
        batch_size = batch_latent_input.shape[0]
        target_len = batch_latent_labels.shape[1]
        latent_dim = batch_latent_labels.shape[2]
        # recon forward
        batch_recon_pred = conv_ae.decoder(batch_latent_pred.reshape(batch_size * target_len, latent_dim)).detach()
        batch_recon_pred = batch_recon_pred.reshape(batch_size, target_len, 3, 128, 128)
        # compute recon psnr
        batch_recon_pred = batch_recon_pred.detach().cpu()
        batch_recon_labels = batch_recon_labels.cpu()
        print(f"batch_recon_pred shape: {batch_recon_pred.shape}")
        psnr_value = peak_signal_noise_ratio(batch_recon_pred, batch_recon_labels)
        psnr_values.append(psnr_value.unsqueeze(0))
        if rollout == False:
            print(f"batch_recon_pred $ batch_recon_labels psnr: {psnr_value}")
        else:
            print(f"rollout batch_recon_pred $ batch_recon_labels psnr: {psnr_value}")
    if rollout == False:
        print(f"Total psnr values: {psnr_values}")
        psnr_values_tensor = torch.cat(psnr_values)
        psnr_values_mean = torch.mean(psnr_values_tensor)
        print(f"Total psnr mean: {psnr_values_mean}")
    else:
        print(f"rollout Total psnr values: {psnr_values}")
        psnr_values_tensor = torch.cat(psnr_values)
        psnr_values_mean = torch.mean(psnr_values_tensor)
        print(f"rollout Total psnr mean: {psnr_values_mean}")


if __name__ == '__main__':
    '''
    dataset_params for recon test:
        test_batch_size for mae: 2292
        test_batch_size foe ssim & psnr: 764
        test_batch_size for rollout mae: 2232
        test_batch_size foe rollout ssim & psnr: 744
        test_num_workers: 1
    '''
    # configuration
    device = torch.device('cuda:5')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    # model name
    recon_type, pred_type = "ae", "lstm"
    recon_number, pred_number = "1", "1"
    conv_ae_name, pred_model_name = f"{recon_type}_{recon_number}", f"{pred_type}_{pred_number}"
    print(f"{conv_ae_name} {pred_model_name}")
    # load saved conv_ae
    conv_ae_dir = f"./saved_models/latent128/{conv_ae_name}/"
    conv_ae_best_epoch = torch.load(conv_ae_dir + f"{recon_type}_best_epoch.pt")
    print(f"{conv_ae_name} best epoch: {conv_ae_best_epoch}")
    conv_ae, _ = init_recon_model()
    conv_ae.load_state_dict(torch.load(conv_ae_dir + f"{recon_type}_best.pt", map_location='cpu'))
    conv_ae = conv_ae.to(device)
    # load saved lstm
    isConditional = True if pred_type == "conditional_lstm" else False
    lstm_dir = f"./saved_models/latent128/{pred_model_name}/{conv_ae_name}/"
    lstm_best_epoch = torch.load(lstm_dir + f"{pred_type}_best_epoch.pt")
    print(f"{pred_model_name} best epoch: {lstm_best_epoch}")
    lstm = init_pred_model(isConditional)
    lstm.load_state_dict(torch.load(lstm_dir + f"{pred_type}_best.pt", map_location='cpu'))
    lstm = lstm.to(device)
    # load dataset
    latent_standard_dataset, latent_standard_dataloader = init_latent_pred_data(config, "test", conv_ae_name, shuffle=False, rollout=False)
    latent_rollout_dataset, latent_rollout_dataloader = init_latent_pred_data(config, "test", conv_ae_name, shuffle=False, rollout=True)
    recon_standard_dataset, recon_standard_dataloader = init_pred_data(config, "test", rollout=False)
    recon_rollout_dataset, recon_rollout_dataloader = init_pred_data(config, "test", rollout=True)

    # compute metrics for standard pred
    mae(conv_ae, lstm, recon_standard_dataloader, latent_standard_dataloader, isConditional, rollout=False)
    # ssim(conv_ae, lstm, recon_standard_dataloader, latent_standard_dataloader, isConditional, rollout=False)
    # psnr(conv_ae, lstm, recon_standard_dataloader, latent_standard_dataloader, isConditional, rollout=False)

    # compute metrics for rollout pred
    # mae(conv_ae, lstm, recon_standard_dataloader, latent_standard_dataloader, isConditional, rollout=True)
    # ssim(conv_ae, lstm, recon_standard_dataloader, latent_standard_dataloader, isConditional, rollout=True)
    # psnr(conv_ae, lstm, recon_standard_dataloader, latent_standard_dataloader, isConditional, rollout=True)
