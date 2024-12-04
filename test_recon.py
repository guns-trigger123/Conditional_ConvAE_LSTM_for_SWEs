import torch
import yaml
from matplotlib import pyplot as plt
from utils import init_recon_model, init_recon_data
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torcheval.metrics.functional import peak_signal_noise_ratio


def mae(conv_ae, test_dataloader):
    # turn to eval mode
    conv_ae.eval()
    for iter, batch in enumerate(test_dataloader):
        # input & recon
        batch_input = batch["input"].to(device)
        batch_recon = conv_ae(batch_input)
        # compute relative error
        batch_recon = batch_recon.detach().cpu()
        batch_labels = batch_input.cpu()
        batch_err = torch.abs(batch_labels - batch_recon)
        batch_rela_err = batch_err / (1 + batch_labels)
        uvh_mean_err = torch.mean(batch_rela_err, dim=(0, 2, 3))
        print(f"relative uvh mean error: {uvh_mean_err.numpy()}")
        # plot one sample
        recon = batch_recon[0].numpy()
        labels = batch_labels[0].numpy()
        err = batch_err[0].numpy()
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        titles = ['u', 'v', 'h']
        for i in range(3):
            im_recon, im_labels, im_err = axes[0][i].imshow(recon[i]), axes[1][i].imshow(labels[i]), axes[2][i].imshow(err[i])
            axes[0][i].set_title(titles[i]), axes[1][i].set_title(titles[i]), axes[2][i].set_title(titles[i])
            axes[0][i].axis('off'), axes[1][i].axis('off'), axes[2][i].axis('off')
            fig.colorbar(im_recon, ax=axes[0][i], orientation='vertical')
            fig.colorbar(im_labels, ax=axes[1][i], orientation='vertical')
            fig.colorbar(im_err, ax=axes[2][i], orientation='vertical')
        # plt.show()


def ssim(conv_ae, test_dataloader):
    # turn to eval mode
    conv_ae.eval()
    # init ssim_calculator
    ssim_calculator = StructuralSimilarityIndexMeasure(data_range=1.0)

    ssim_values_list = []
    for iter, batch in enumerate(test_dataloader):
        # input & recon
        batch_input = batch["input"].to(device)
        batch_recon = conv_ae(batch_input)
        # compute ssim
        batch_recon = batch_recon.detach().cpu()
        batch_labels = batch_input.cpu()
        print(f"batch_recon shape: {batch_recon.shape}")
        batch_ssim = ssim_calculator(batch_recon, batch_labels)
        ssim_values_list.append(batch_ssim.unsqueeze(0))
        print(f"batch ssim: {batch_ssim}")

    print(f"ssim values list: {ssim_values_list}")
    ssim_values_tensor = torch.cat(ssim_values_list)
    ssim_values_mean = torch.mean(ssim_values_tensor)
    print(f"ssim mean: {ssim_values_mean}")


def psnr(conv_ae, test_dataloader):
    # turn to eval mode
    conv_ae.eval()

    psnr_values_list = []
    for iter, batch in enumerate(test_dataloader):
        # input & recon
        batch_input = batch["input"].to(device)
        batch_recon = conv_ae(batch_input)
        # compute psnr
        batch_recon = batch_recon.detach().cpu()
        batch_labels = batch_input.cpu()
        print(f"batch_recon shape: {batch_recon.shape}")
        batch_psnr = peak_signal_noise_ratio(batch_recon, batch_labels)
        psnr_values_list.append(batch_psnr.unsqueeze(0))
        print(f"batch psnr: {batch_psnr}")

    print(f"psnr values list: {psnr_values_list}")
    psnr_values_tensor = torch.cat(psnr_values_list)
    psnr_values_mean = torch.mean(psnr_values_tensor)
    print(f"psnr mean: {psnr_values_mean}")


if __name__ == '__main__':
    '''
    dataset_params for recon test:
        test_batch_size for mae: 2400
        test_batch_size foe ssim & psnr: 400
        test_num_workers: 1
    '''
    # configuration
    device = torch.device('cuda:0')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    # load dataset
    test_dataset, test_dataloader = init_recon_data(config, "test")
    # load saved model
    SAVED_DIRECTORY = f"./saved_models/latent128/conditional_ae_3/"
    SAVED_PREFIX = "conditional_ae"
    best_epoch = torch.load(SAVED_DIRECTORY + f"{SAVED_PREFIX}_best_epoch.pt")
    print(f"best epoch: {best_epoch}")
    conv_ae, _ = init_recon_model()
    conv_ae.load_state_dict(torch.load(SAVED_DIRECTORY + f"{SAVED_PREFIX}_best.pt", map_location='cpu'))
    conv_ae = conv_ae.to(device)
    # compute metrics
    mae(conv_ae, test_dataloader)
    # ssim(conv_ae, test_dataloader)
    # psnr(conv_ae, test_dataloader)
