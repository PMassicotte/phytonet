"""Advanced data augmentation techniques for improving generalization."""

import torch
import torch.nn as nn
import numpy as np


class MixUp:
    """
    MixUp data augmentation.
    
    Reference: Zhang, H., et al. (2017). mixup: Beyond empirical risk minimization.
    arXiv preprint arXiv:1710.09412.
    
    Args:
        alpha: Beta distribution parameter (default: 1.0)
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, x, y):
        """
        Apply MixUp augmentation.
        
        Args:
            x: Input batch (batch_size, channels, height, width)
            y: Target labels (batch_size,)
            
        Returns:
            Mixed inputs, original targets, shuffled targets, mixing lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """
    CutMix data augmentation.
    
    Reference: Yun, S., et al. (2019). CutMix: Regularization strategy to train 
    strong classifiers with localizable features. ICCV 2019.
    
    Args:
        alpha: Beta distribution parameter (default: 1.0)
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, x, y):
        """
        Apply CutMix augmentation.
        
        Args:
            x: Input batch (batch_size, channels, height, width)  
            y: Target labels (batch_size,)
            
        Returns:
            Mixed inputs, original targets, shuffled targets, mixing lambda
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        y_a, y_b = y, y[index]
        
        # Generate random bounding box
        W = x.size(3)
        H = x.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        return x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute loss for MixUp/CutMix augmented samples.
    
    Args:
        criterion: Loss function
        pred: Model predictions
        y_a: Original targets
        y_b: Shuffled targets  
        lam: Mixing parameter
        
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class RandomAugmentationStrategy:
    """
    Randomly apply MixUp, CutMix, or no augmentation during training.
    
    Args:
        mixup_prob: Probability of applying MixUp (default: 0.3)
        cutmix_prob: Probability of applying CutMix (default: 0.3)
        mixup_alpha: MixUp alpha parameter (default: 1.0)
        cutmix_alpha: CutMix alpha parameter (default: 1.0)
    """
    
    def __init__(self, mixup_prob=0.3, cutmix_prob=0.3, mixup_alpha=1.0, cutmix_alpha=1.0):
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.mixup = MixUp(alpha=mixup_alpha)
        self.cutmix = CutMix(alpha=cutmix_alpha)
        
    def __call__(self, x, y):
        """
        Randomly apply augmentation strategy.
        
        Returns:
            Augmented inputs, targets, and mixing info
        """
        rand = np.random.random()
        
        if rand < self.mixup_prob:
            return self.mixup(x, y) + ("mixup",)
        elif rand < self.mixup_prob + self.cutmix_prob:
            return self.cutmix(x, y) + ("cutmix",)
        else:
            return x, y, y, 1.0, "none"