import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CE_KL(nn.Module):
    """
    compute the crossentropy loss and KL loss
    """
    def __init__(self):
        super(CE_KL, self).__init__()
        self.ce = nn.CrossEntropyLoss().to(device)

    def forward(self, logit, mu, sigma2, cap, cap_len):
        target = cap[:, 1:]
        cap_len = cap_len - 1

        target = pack_padded_sequence(target, cap_len, batch_first=True, enforce_sorted=False)[0]
        logit = pack_padded_sequence(logit, cap_len, batch_first=True, enforce_sorted=False)[0]

        # reconstruct loss
        loss_ce = self.ce(logit, target)

        # KL-divergence (entire batch)
        loss_kl = ((-0.5 * torch.sum(1 + sigma2 - torch.exp(sigma2) - mu**2))/mu.size(0))  # KL散度的计算

        return loss_ce, loss_kl  # 目标是最小化1.重建损失 2.后验和标准正态的距离
