import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeAlignmentLoss(nn.Module):
    """Alignment loss with hard-negative penalty."""
    def __init__(self, temperature=0.07, hardneg_weight=0.5, topk=5, margin=0.0):
        super(HardNegativeAlignmentLoss, self).__init__()
        self.temperature = temperature
        self.hardneg_weight = hardneg_weight
        self.topk = topk
        self.margin = margin

    def _hard_negative_penalty(self, logits):
        """Symmetric hard-negative penalty (t->l and l->t)."""
        B = logits.size(0)
        device = logits.device
        diag = logits.diag().view(B, 1)
        mask = torch.eye(B, device=device).bool()
        logits_neg = logits.masked_fill(mask, float('-inf'))
        topk_vals_t2l, _ = torch.topk(logits_neg, k=min(self.topk, max(1, B-1)), dim=1)
        penalty_t2l = F.relu(topk_vals_t2l - diag + self.margin).mean()
        logits_T = logits.t()
        diag_T = logits_T.diag().view(B, 1)
        logits_neg_T = logits_T.masked_fill(mask, float('-inf'))
        topk_vals_l2t, _ = torch.topk(logits_neg_T, k=min(self.topk, max(1, B-1)), dim=1)
        penalty_l2t = F.relu(topk_vals_l2t - diag_T + self.margin).mean()

        return (penalty_t2l + penalty_l2t) * 0.5

    def forward(self, text_embeddings, label_embeddings):
        t = F.normalize(text_embeddings, dim=-1)
        l = F.normalize(label_embeddings, dim=-1)
        logits = torch.matmul(t, l.t()) / self.temperature

        B = t.size(0)
        labels = torch.arange(B, device=t.device)
        loss_t2l = F.cross_entropy(logits, labels)
        loss_l2t = F.cross_entropy(logits.t(), labels)
        base = (loss_t2l + loss_l2t) * 0.5

        hardneg = self._hard_negative_penalty(logits)
        return base + self.hardneg_weight * hardneg



def create_loss_function(args):
    """Create alignment loss function."""
    return HardNegativeAlignmentLoss(
        temperature=args.temperature,
        hardneg_weight=getattr(args, 'hardneg_weight', 0.5),
        topk=getattr(args, 'hardneg_topk', 5),
        margin=0.4
    )