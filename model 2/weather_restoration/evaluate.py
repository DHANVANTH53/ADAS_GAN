 # evaluate.py
import os
import torch
from models.unet import UNet
from utils.dataset import get_dataloader
from utils.metrics import compute_psnr, compute_ssim
from PIL import Image
import torchvision.transforms as T
import numpy as np

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    dl = get_dataloader(args.data_root, split='test', img_size=args.img_size, batch_size=1, shuffle=False, num_workers=0)
    os.makedirs(args.out_dir, exist_ok=True)

    psnrs, ssims = [], []
    transform = T.ToPILImage()
    with torch.no_grad():
        for i, (rainy, clean) in enumerate(dl):
            rainy = rainy.to(device)
            clean = clean.to(device)
            pred = model(rainy)
            psnrs.append(compute_psnr(pred, clean))
            ssims.append(compute_ssim(pred, clean))

            # save sample comparison
            inp = transform(rainy.squeeze(0).cpu())
            out = transform(pred.squeeze(0).cpu())
            gt = transform(clean.squeeze(0).cpu())
            concat = np.concatenate([np.array(inp), np.array(out), np.array(gt)], axis=1)
            Image.fromarray(concat).save(os.path.join(args.out_dir, f'sample_{i}.png'))
    print(f"Mean PSNR: {sum(psnrs)/len(psnrs):.3f}, Mean SSIM: {sum(ssims)/len(ssims):.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/best_ckpt.pth')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--out_dir', type=str, default='eval_out')
    args = parser.parse_args()
    evaluate(args)

