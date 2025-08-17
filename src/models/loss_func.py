import torch
import torch.nn.functional as F
import torch.nn as nn
class CombinedSignalLoss(nn.Module):
    """
    Composite loss combining MAE, MSE, and (1 - Pearson Correlation).

    Parameters:
        mae_weight (float): Weight of Mean Absolute Error.
        mse_weight (float): Weight of Mean Squared Error.
        pcc_weight (float): Weight of 1 - Pearson Correlation.
        eps (float): Small value to avoid division by zero in PCC.
    """
    def __init__(self,mae_weight=1.0, mse_weight=1.0, pcc_weight=1.0, eps=1e-8):
        super().__init__()
        self.mae_weight = mae_weight
        self.mse_weight = mse_weight
        self.pcc_weight = pcc_weight
        self.eps = eps
    def pearson_correlation(self, x, y):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        return torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx ** 2) + self.eps) *
            torch.sqrt(torch.sum(vy ** 2) + self.eps)
        )
    def forward(self, predictions, targets):
        # Ensure input is [B, L]
        if predictions.dim() == 3:
            predictions = predictions.squeeze(1)
            targets = targets.squeeze(1)

        mae_loss = F.l1_loss(predictions, targets)
        mse_loss = F.mse_loss(predictions, targets)
        pcc_loss = 1.0 - self.pearson_correlation(predictions, targets)

        total_loss = (
            self.mae_weight * mae_loss +
            self.mse_weight * mse_loss +
            self.pcc_weight * pcc_loss
        )
        return total_loss