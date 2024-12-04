import os
import time
import torch
import yaml
import warnings
import logging
import numpy as np
import torch.optim as optim
from utils import init_pred_model, init_latent_pred_data, load_pred_checkpoint, load_pred_best_epoch

if __name__ == '__main__':
    '''
    dataset_params for recon training:
        train_batch_size: 50
        val_batch_size: 3820
        train_num_workers: 3
        val_num_workers: 1
    '''
    # configuration
    device = torch.device('cuda:0')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    # init model
    recon_type, pred_type = config['logging_params']['recon_type'], config['logging_params']['pred_type']
    recon_number, pred_number = config['logging_params']['recon_number'], config['logging_params']['pred_number']
    conv_ae_name, pred_model_name = f"{recon_type}_{recon_number}", f"{pred_type}_{pred_number}"
    isConditional = True if pred_type == "conditional_lstm" else False
    lstm = init_pred_model(isConditional)
    lstm = lstm.to(device)
    # load dataset
    train_dataset, train_dataloader = init_latent_pred_data(config, "train", conv_ae_name)
    val_dataset, val_dataloader = init_latent_pred_data(config, "val", conv_ae_name)
    # init optimizer and lr_scheduler
    opt = optim.Adam(lstm.parameters(), lr=config["experiment_params"]["lstm_lr"])
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config["experiment_params"]["sch_T_0"], T_mult=config["experiment_params"]["sch_T_mult"])
    # init loss function / criterion
    criterion = torch.nn.L1Loss()
    # init max iterations and epochs
    MAX_ITER, MAX_VAL_ITER, MAX_EPOCH = len(train_dataloader), len(val_dataloader), config["experiment_params"]["max_epochs"]
    # init logging names
    SAVED_DIRECTORY = f"./saved_models/latent128/{pred_model_name}/{conv_ae_name}/"
    SAVED_PREFIX = config["logging_params"]["pred_type"]
    logging.basicConfig(
        filename=f"./loss/latent128/{pred_model_name}/{conv_ae_name}/loss.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s)"
    )
    # log configuration
    with open(SAVED_DIRECTORY + 'train_pred_config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    # train from scratch (PRE_TRAIN = 0) / train at checkpoint (PRE_TRAIN = 1)
    PRE_TRAIN = 0
    if PRE_TRAIN == 0:
        best_epoch, last_epoch = -1, 0
        min_err = np.array([100, 100, 100, 100, 100], dtype=np.float32)
    else:
        lstm, opt, sch, last_epoch = load_pred_checkpoint(SAVED_DIRECTORY, SAVED_PREFIX, lstm, opt, sch)
        best_epoch, min_err = load_pred_best_epoch(SAVED_DIRECTORY, SAVED_PREFIX, lstm, device, val_dataloader, isConditional)
    # main training process
    warnings.simplefilter("ignore", category=UserWarning)
    for epoch in range(last_epoch + 1, last_epoch + 1 + MAX_EPOCH):
        # training
        t1 = time.time()
        # turn to train mode
        lstm.train()
        for iter, batch in enumerate(train_dataloader):
            # input & labels
            batch_input, batch_target = batch["input"].to(device), batch["target"].to(device)
            # forward & loss
            if isConditional:
                R_0, Hp_0 = batch["R"].reshape(-1, 1, 1) / 40.0, batch["Hp"].reshape(-1, 1, 1) / 20.0
                R, Hp = R_0.repeat(1, 5, 1), Hp_0.repeat(1, 5, 1)
                batch_concate = torch.cat([R, Hp], dim=2).to(device)
                batch_sup_weight = Hp_0.squeeze(2).to(device)
                batch_input = torch.cat([batch_input, batch_concate], dim=2)
                batch_pred = lstm(batch_input, batch_sup_weight)
            else:
                batch_pred = lstm(batch_input)
            loss = criterion(batch_pred, batch_target)
            # optimizer update
            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()

            if (iter + 1) % MAX_ITER == 0:
                # log training loss
                logging.info(f"epoch: {epoch} iter: {iter} " + f"pred latent training_loss: {loss}")
                # save lstm
                save_path_lstm = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + f'_{epoch}_{iter + 1}.pt')
                torch.save(lstm.state_dict(), save_path_lstm)
                # save checkpoint
                CHECKPOINT_NAME = "checkpoint_" + SAVED_PREFIX + ".pt"
                save_path_checkpoint = os.path.join(SAVED_DIRECTORY, CHECKPOINT_NAME)
                torch.save({
                    'epoch': epoch,
                    'lstm': lstm.state_dict(),
                    'opt': opt.state_dict(),
                    'sch': sch.state_dict(),
                }, save_path_checkpoint)
        t2 = time.time()
        print(f"epoch: {epoch} latent pred train time: {t2 - t1}s ")
        # validation
        # turn to val mode
        lstm.eval()
        for iter, batch in enumerate(val_dataloader):
            batch_input, batch_target = batch["input"].to(device), batch["target"].to(device)
            if isConditional:
                R_0, Hp_0 = batch["R"].reshape(-1, 1, 1) / 40.0, batch["Hp"].reshape(-1, 1, 1) / 20.0
                R, Hp = R_0.repeat(1, 5, 1), Hp_0.repeat(1, 5, 1)
                batch_concate = torch.cat([R, Hp], dim=2).to(device)
                batch_sup_weight = Hp_0.squeeze(2).to(device)
                batch_input = torch.cat([batch_input, batch_concate], dim=2)
                batch_pred = lstm(batch_input, batch_sup_weight)
            else:
                batch_pred = lstm(batch_input)

            batch_err = torch.abs(batch_target - batch_pred)
            uvh_mean_err = torch.mean(batch_err, dim=(0, 2))
            uvh_mean_err_np = uvh_mean_err.cpu().detach().numpy()

            if np.sum(uvh_mean_err_np) < np.sum(min_err):
                best_epoch = epoch
                min_err = uvh_mean_err_np
                save_path_best_lstm = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + '_best.pt')
                torch.save(lstm.state_dict(), save_path_best_lstm)
                save_path_best_epoch = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + '_best_epoch.pt')
                torch.save(torch.tensor([best_epoch]), save_path_best_epoch)
                print(f"best_epoch: {best_epoch}")
        t3 = time.time()
        print(f"epoch: {epoch} latent pred val time: {t3 - t2}s ")
    print(f"Total Best Epoch: {best_epoch}")
