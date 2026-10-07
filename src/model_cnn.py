"""Experiment 2 model: ResNet18 transfer learning, regression head.

ResNet18 -> global average pool (built-in) -> Linear(512, 1) -> diameter_nm.

Supports:
  - 4-channel input (image x3 + scale channel) by replacing the stem conv
  - 'frozen' vs 'finetune' backbone modes (Experiment 3 ablation)
  - MC-Dropout: a Dropout layer before the regression head that is kept
    ACTIVE at inference time (model.train() during prediction) to produce an
    ensemble of stochastic predictions -> mean = prediction, std = uncertainty
"""
import torch
import torch.nn as nn
import torchvision.models as models


class NanoMeasureNet(nn.Module):
    def __init__(self, in_channels=3, backbone_mode="finetune", dropout_p=0.3):
        super().__init__()
        assert backbone_mode in ("frozen", "partial", "finetune")
        self.backbone_mode = backbone_mode

        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.pretrained = True
        except Exception as e:
            # This sandbox's network policy blocks download.pytorch.org (and
            # other external model-weight hosts), so ImageNet-pretrained
            # weights cannot be fetched here. Fall back to random init rather
            # than silently failing or claiming transfer learning that didn't
            # happen -- see README "What I would NOT claim".
            print(f"  [model_cnn] Could not fetch pretrained ResNet18 weights "
                  f"({type(e).__name__}); using random initialization instead.")
            resnet = models.resnet18(weights=None)
            self.pretrained = False

        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels,
                                   kernel_size=old_conv.kernel_size,
                                   stride=old_conv.stride, padding=old_conv.padding,
                                   bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                if in_channels > 3:
                    new_conv.weight[:, 3:] = old_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1)
            resnet.conv1 = new_conv

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # drop final FC, keep avgpool
        self.dropout = nn.Dropout(p=dropout_p)
        self.regressor = nn.Linear(resnet.fc.in_features, 1)

        self._apply_backbone_mode()

    def _apply_backbone_mode(self):
        if self.backbone_mode == "frozen":
            for p in self.features.parameters():
                p.requires_grad = False
        elif self.backbone_mode == "partial":
            # freeze everything except the last residual block (layer4)
            for name, p in self.features.named_parameters():
                p.requires_grad = "7." in name  # layer4 is index 7 in this Sequential
        # "finetune": all params trainable (default requires_grad=True)

    def forward(self, x):
        feat = self.features(x)             # (B, 512, 1, 1)
        feat = torch.flatten(feat, 1)        # (B, 512)
        feat = self.dropout(feat)
        out = self.regressor(feat)           # (B, 1)
        return out.squeeze(-1)

    def get_last_conv_layer(self):
        """For Grad-CAM: the last conv block (layer4) before global pooling.
        self.features = [conv1, bn1, relu, maxpool, layer1, layer2, layer3,
        layer4, avgpool] -> layer4 is index -2."""
        return self.features[-2]
