import os
import torch
import yaml
import time
import numpy as np
import torch.optim as optim
import logging
from utils import init_recon_model, init_recon_data, load_recon_checkpoint, load_recon_best_epoch

if __name__ == '__main__':
    '''
    dataset_params for recon training:
        train_batch_size: 50
        val_batch_size: 4000
        train_num_workers: 3
        val_num_workers: 1
    '''
    # configuration
    device = torch.device('cuda:1')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    # load dataset
    train_dataset, train_dataloader = init_recon_data(config, "train")
    val_dataset, val_dataloader = init_recon_data(config, "val")
    # init model
    conv_ae, mlp = init_recon_model()
    conv_ae, mlp = conv_ae.to(device), mlp.to(device)
    # init optimizer and lr_scheduler
    opt_conv_ae = optim.Adam(conv_ae.parameters(), lr=config["experiment_params"]["conv_ae_lr"])
    opt_mlp = optim.Adam(mlp.parameters(), lr=config["experiment_params"]["mlp_lr"])
    sch_conv_ae = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_conv_ae, T_0=config["experiment_params"]["sch_T_0"], T_mult=config["experiment_params"]["sch_T_mult"])
    sch_mlp = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_mlp, T_0=config["experiment_params"]["sch_T_0"], T_mult=config["experiment_params"]["sch_T_mult"])
    # init loss function / criterion
    criterion = torch.nn.L1Loss()
    # init max iterations and epochs
    MAX_ITER, MAX_VAL_ITER, MAX_EPOCH = len(train_dataloader), len(val_dataloader), config["experiment_params"]["max_epochs"]
    # init logging names
    SAVED_DIRECTORY = f"./saved_models/latent128/{config['logging_params']['recon_type']}_{config['logging_params']['recon_number']}/"
    SAVED_PREFIX = config["logging_params"]["recon_type"]
    logging.basicConfig(
        filename=f"./loss/latent128/{config['logging_params']['recon_type']}_{config['logging_params']['recon_number']}/loss.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s)"
    )
    # log configuration
    with open(SAVED_DIRECTORY + 'train_recon_config.yaml', 'w') as f:
        yaml.safe_dump(config, f)
    # train from scratch (PRE_TRAIN = 0) / train at checkpoint (PRE_TRAIN = 1)
    PRE_TRAIN = 0
    if PRE_TRAIN == 0:
        last_epoch, best_epoch = -1, -1
        min_err = np.array([100, 100, 100], dtype=np.float32)
    else:
        conv_ae, opt_conv_ae, sch_conv_ae, mlp, opt_mlp, sch_mlp, last_epoch = load_recon_checkpoint(
            SAVED_DIRECTORY, SAVED_PREFIX,
            conv_ae, opt_conv_ae, sch_conv_ae,
            mlp, opt_mlp, sch_mlp
        )
        best_epoch, min_err = load_recon_best_epoch(SAVED_DIRECTORY, SAVED_PREFIX, conv_ae, device, val_dataloader)
    # turn to train mode
    conv_ae.train()
    mlp.train()
    # main training process
    for epoch in range(last_epoch + 1, last_epoch + 1 + MAX_EPOCH):
        # training
        t1 = time.time()
        for iter, batch in enumerate(train_dataloader):
            # input & labels
            batch_input = batch["input"].to(device)
            R, Hp = batch["R"].reshape(-1, 1) / 40.0, batch["Hp"].reshape(-1, 1) / 20.0
            reg_labels = torch.cat([R, Hp], dim=1).float().to(device)
            # forward & loss
            batch_recon, batch_latent = conv_ae(batch_input), conv_ae.encoder(batch_input)
            reg_recon = mlp(batch_latent)
            loss_recon, loss_reg = criterion(batch_recon, batch_input), criterion(reg_recon, reg_labels)
            loss_reg_weight = config["experiment_params"]["mlp_weight"]
            if config["experiment_params"]["mlp_weight_mode"] == "gradual":
                if epoch <= 1000:
                    loss_reg_weight = 0.0
                else:
                    # loss2_weight = loss2_weight * (epoch - 1000) / MAX_EPOCH
                    loss_reg_weight = loss_reg_weight
            loss = loss_recon + loss_reg_weight * loss_reg
            # optimizer update
            opt_conv_ae.zero_grad(), opt_mlp.zero_grad()
            loss.backward()
            opt_conv_ae.step(), opt_mlp.step()
            sch_conv_ae.step(), sch_mlp.step()

            if (iter + 1) % MAX_ITER == 0:
                # log training loss
                logging.info(f"epoch: {epoch} iter: {iter} " +
                             f"training_loss: {loss} loss_recon:{loss_recon} loss_reg:{loss_reg} loss_reg_weight:{loss_reg_weight}")
                # save conv_ae
                save_path_conv_ae = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + f'_{epoch}_{iter + 1}.pt')
                torch.save(conv_ae.state_dict(), save_path_conv_ae)
                # save mlp
                save_path_mlp = os.path.join(SAVED_DIRECTORY, 'mlp' + f'_{epoch}_{iter + 1}.pt')
                torch.save(mlp.state_dict(), save_path_mlp)
                # save checkpoint
                CHECKPOINT_NAME = "checkpoint_" + SAVED_PREFIX + ".pt"
                save_path_checkpoint = os.path.join(SAVED_DIRECTORY, CHECKPOINT_NAME)
                torch.save({
                    'epoch': epoch,
                    'conv_ae': conv_ae.state_dict(),
                    'mlp': mlp.state_dict(),
                    'opt_conv_ae': opt_conv_ae.state_dict(),
                    'opt_mlp': opt_mlp.state_dict(),
                    'sch_conv_ae': sch_conv_ae.state_dict(),
                    'sch_mlp': sch_mlp.state_dict(),
                }, save_path_checkpoint)
        t2 = time.time()
        print(f"epoch: {epoch} train time: {t2 - t1}s ")
        # validation
        uvh_mean_rela_err_list = []
        for iter, batch in enumerate(val_dataloader):
            batch_input = batch["input"].to(device)
            batch_recon = conv_ae(batch_input)
            batch_err = torch.abs(batch_input - batch_recon)
            batch_rela_err = batch_err / (1 + batch_input)

            uvh_mean_rela_err = torch.mean(batch_rela_err, dim=(0, 2, 3))
            uvh_mean_rela_err_np = uvh_mean_rela_err.cpu().detach().numpy()
            uvh_mean_rela_err_list.append(uvh_mean_rela_err_np)

            if iter == MAX_VAL_ITER - 1:
                uvh_list_mean_rela_err = sum(uvh_mean_rela_err_list) / MAX_VAL_ITER
                uvh_list_mean_rela_err_sum = np.sum(uvh_list_mean_rela_err)
                logging.info(f"epoch: {epoch} val uvh_list_mean_rela_err {uvh_list_mean_rela_err}" +
                             f"val uvh_list_mean_rela_err_sum {uvh_list_mean_rela_err_sum}")

                if uvh_list_mean_rela_err_sum < np.sum(min_err):
                    best_epoch = epoch
                    min_err = uvh_list_mean_rela_err
                    save_path_best_conv_ae = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + '_best.pt')
                    torch.save(conv_ae.state_dict(), save_path_best_conv_ae)
                    save_path_best_epoch = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + '_best_epoch.pt')
                    torch.save(torch.tensor([best_epoch]), save_path_best_epoch)
                    print(f"best_epoch: {best_epoch}")
        t3 = time.time()
        print(f"epoch: {epoch} val time: {t3 - t2}s ")
    print(f"Total Best Epoch: {best_epoch}")
