# train.py (UPDATED - supports unet or restormer + AMP)
import os
import torch
from torch import nn, optim
from models.unet import UNet
from models.restormer import RestormerUNet
from utils.dataset import get_dataloader
from losses.perceptual_loss import VGGPerceptualLoss
from losses.ssim_loss import SSIMLoss
from utils.metrics import compute_psnr, compute_ssim
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def save_plots(train_losses, val_losses, val_psnrs, val_ssims, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(val_psnrs, label='val_psnr')
    plt.plot(val_ssims, label='val_ssim')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'val_metrics.png'))
    plt.close()

def build_model(name, device, **kwargs):
    name = name.lower()
    if name == 'unet':
        model = UNet(in_channels=3, out_channels=3)
    elif name == 'restormer':
        model = RestormerUNet(in_channels=3, out_channels=3,
                              features=kwargs.get('features',[64,128,256,512]),
                              transformer_dim=kwargs.get('transformer_dim',512),
                              patch_size=kwargs.get('patch_size',8),
                              nhead=kwargs.get('nhead',8),
                              num_transformer_layers=kwargs.get('num_layers',8),
                              ff_dim=kwargs.get('ff_dim',2048))
    else:
        raise ValueError("Unknown model: choose 'unet' or 'restormer'")
    return model.to(device)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # build model
    model = build_model(args.model, device,
                        features=[64,128,256,512],
                        transformer_dim=args.transformer_dim,
                        patch_size=args.patch_size,
                        nhead=args.nhead,
                        num_layers=args.num_layers,
                        ff_dim=args.ff_dim)
    print("Model params:", sum(p.numel() for p in model.parameters())/1e6, "M")

    train_loader = get_dataloader(args.data_root, split='train', img_size=args.img_size, batch_size=args.batch_size)
    val_loader = get_dataloader(args.data_root, split='val', img_size=args.img_size, batch_size=args.batch_size, shuffle=False)

    l1 = nn.L1Loss()
    perc = VGGPerceptualLoss().to(device)
    ssim_loss = SSIMLoss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

    best_psnr = 0.0
    train_losses, val_losses, val_psnrs, val_ssims = [], [], [], []

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    for epoch in range(1, args.epochs+1):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for rainy, clean in loop:
            rainy = rainy.to(device)
            clean = clean.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                pred = model(rainy)
                loss_rec = l1(pred, clean)
                loss_perc = perc(pred, clean)
                loss_ssim = ssim_loss(pred, clean)
                loss = args.lambda_rec * loss_rec + args.lambda_perc * loss_perc + args.lambda_ssim * loss_ssim
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # validation
        model.eval()
        val_loss = 0.0
        psnr_vals, ssim_vals = [], []
        with torch.no_grad():
            for rainy, clean in val_loader:
                rainy = rainy.to(device)
                clean = clean.to(device)
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    pred = model(rainy)
                    loss_rec = l1(pred, clean)
                    loss_perc = perc(pred, clean)
                    loss_ssim_v = ssim_loss(pred, clean)
                    loss_v = args.lambda_rec * loss_rec + args.lambda_perc * loss_perc + args.lambda_ssim * loss_ssim_v
                val_loss += loss_v.item()
                psnr_vals.append(compute_psnr(pred, clean))
                ssim_vals.append(compute_ssim(pred, clean))

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        mean_psnr = sum(psnr_vals)/len(psnr_vals) if psnr_vals else 0.0
        mean_ssim = sum(ssim_vals)/len(ssim_vals) if ssim_vals else 0.0
        val_psnrs.append(mean_psnr)
        val_ssims.append(mean_ssim)

        print(f"[Epoch {epoch}] TrainLoss: {avg_train_loss:.4f}, ValLoss: {avg_val_loss:.4f}, ValPSNR: {mean_psnr:.3f}, ValSSIM: {mean_ssim:.4f}")

        # save best
        os.makedirs(args.out_dir, exist_ok=True)
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr
            }
            torch.save(ckpt, os.path.join(args.out_dir, 'best_ckpt.pth'))
            print(f"Saved best checkpoint at epoch {epoch} with PSNR {best_psnr:.3f}")

        # save periodic checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict()
            }, os.path.join(args.out_dir, f'ckpt_epoch_{epoch}.pth'))

        save_plots(train_losses, val_losses, val_psnrs, val_ssims, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='restormer', choices=['unet','restormer'])
    parser.add_argument('--data_root', type=str, default='data', help='root dataset dir')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argu_
