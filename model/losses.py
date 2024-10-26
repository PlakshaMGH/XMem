import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict


def dice_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:,i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt==(i+1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()


# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm if start_warm is not None else -1
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])

    def compute(self, data, num_objects, it):
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2]

        losses['total_loss'] = 0
        losses['total_dice_loss'] = 0
        losses['total_ce_loss'] = 0
        for ti in range(1, t): # 0th frame has gt_mask, so we skip it
            for bi in range(b):
                loss, p = self.bce(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0], it)
                losses['p'] += p / b / (t-1)
                losses[f'ce_loss_{ti}'] += loss / b
                losses['total_ce_loss'] += loss / b

            # Add cross entropy loss to total loss
            losses['total_loss'] += losses['ce_loss_%d'%ti]

            losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:,ti,0])
            losses['total_dice_loss'] += losses[f'dice_loss_{ti}']

            # Add dice loss to total loss
            losses['total_loss'] += losses[f'dice_loss_{ti}']

        return losses

    def compute_test(self, pred_masks, gt_masks):
        losses = defaultdict(float)
        t, c, h, w = pred_masks.shape

        for ti in range(t):
            # Compute BCE loss
            bce_loss, _ = self.bce(pred_masks[ti].unsqueeze(0), gt_masks[ti].argmax(dim=0).unsqueeze(0), 0)  # Use iteration 0 for testing
            losses['bce_loss'] += bce_loss.item()

            # Compute Dice loss
            dice_loss_val = dice_loss(pred_masks[ti].unsqueeze(0), gt_masks[ti].argmax(dim=0).unsqueeze(0))
            losses['dice_loss'] += dice_loss_val.item()

        losses['average_bce_loss'] = losses['bce_loss'] / t
        losses['average_dice_loss'] = losses['dice_loss'] / t
        losses['total_avg_loss'] = losses['average_bce_loss'] + losses['average_dice_loss']

        return losses
