import torch.nn as nn

from utils.registry import Registry


MODELS = Registry("models")

def build_model(cfg):
    return MODELS.build(cfg)

@MODELS.register_module("Estimator")
class Estimator(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)

    def forward(self, input_dict):
        pred_exp = self.backbone(input_dict)
        return dict(pred_exp=pred_exp)
