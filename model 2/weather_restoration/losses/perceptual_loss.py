# losses/perceptual_loss.py
import torch
import torch.nn as nn
import torchvision.models as models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.blocks = nn.ModuleList()
        self.blocks.append(vgg[:4].eval())   # relu1_2
        self.blocks.append(vgg[4:9].eval())  # relu2_2
        self.blocks.append(vgg[9:16].eval()) # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.transform = None
        self.resize = resize

    def forward(self, input, target):
        # expect input/target in range [0,1] -> convert to 0..1 normalized to imagenet style
        # Normalize into VGG expected range
        mean = torch.tensor([0.485, 0.456, 0.406]).to(input.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(input.device).view(1,3,1,1)
        input_v = (input - mean) / std
        target_v = (target - mean) / std
        loss = 0.0
        x = input_v
        y = target_v
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss
 
