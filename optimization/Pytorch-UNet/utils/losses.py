import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for handling class imbalance."""

    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets, num_classes=3):
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        tp = (inputs * targets_one_hot).sum(dim=(2, 3))
        fp = (inputs * (1 - targets_one_hot)).sum(dim=(2, 3))
        fn = ((1 - inputs) * targets_one_hot).sum(dim=(2, 3))

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1 - tversky, self.gamma)

        return focal_tversky.mean()


class CombinedLoss(nn.Module):
    """Combined CE + Focal Tversky Loss with class weighting."""

    def __init__(self, ce_weight=0.5, ftl_weight=0.5, class_weights=None):
        super().__init__()
        self.ce_weight = ce_weight
        self.ftl_weight = ftl_weight
        if class_weights is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 3.0]))
        self.ftl = FocalTverskyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        ftl_loss = self.ftl(inputs, targets)
        return self.ce_weight * ce_loss + self.ftl_weight * ftl_loss
