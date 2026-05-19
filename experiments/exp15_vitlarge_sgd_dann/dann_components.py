"""
DANN Components for exp15_vitlarge_sgd_dann
Gradient Reversal Layer + Camera Classifier

FIXES from v1:
- Removed std=0.01 init (near-zero outputs -> cam_acc stuck at 25% from epoch 1)
- Removed Dropout(0.5) (prevented classifier from learning cameras)
- Added BatchNorm for training stability and faster learning
- 3-layer MLP (Aymen said 2-3 layers)
- Progressive lambda handled in processor_dann.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Forward: identity.
    Backward: multiply gradient by -lambda.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        # .item() avoids float16/float32 mismatch with AMP
        return -grad_output * lambda_.item(), None


def grad_reverse(x, lambda_=1.0):
    """Apply gradient reversal with given lambda."""
    lambda_tensor = torch.tensor(lambda_, dtype=torch.float32).to(x.device)
    return GradientReversalFunction.apply(x, lambda_tensor)


class CameraClassifier(nn.Module):
    """
    3-layer MLP camera classifier.
    Input:  (B, 1024) CLS token before bottleneck BN.
    Output: (B, 4) camera logits.

    FC(1024->512)->BN->ReLU -> FC(512->256)->BN->ReLU -> FC(256->4)

    WHY NO Dropout, WHY default init:
      v1 used Dropout(0.5) + std=0.01 -> near-zero logits -> cam_acc=25% from ep1
      Classifier must first LEARN cameras (cam_acc 90%+), THEN GRL fights it down.
      Default kaiming_uniform init ensures fast learning from epoch 1.
    """
    def __init__(self, in_dim=1024, num_cameras=4):
        super(CameraClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_cameras)
        )
        # Default PyTorch kaiming_uniform init - no override needed

    def forward(self, x):
        return self.classifier(x)