 # utils/metrics.py
import torch
import torch.nn.functional as F
import math
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim

def compute_psnr(pred, target):
    # expects tensors in [0,1]
    return psnr(pred, target, data_range=1.0).item()

def compute_ssim(pred, target):
    return ssim(pred, target, data_range=1.0).item()

