# losses/ssim_loss.py
import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim

class SSIMLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # torchmetrics ssim returns mean SSIM across batch
        s = ssim(pred, target, data_range=1.0)  # pred and target in [0,1]
        return 1.0 - s
 
