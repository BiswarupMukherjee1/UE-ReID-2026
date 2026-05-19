"""
DANN Components for exp17_dann_equalized
Gradient Reversal Layer + Camera Classifier

Identical to exp15_vitlarge_sgd_dann/dann_components.py.
Copied here so exp17 is fully self-contained.

Architecture:
  GRL: forward = identity, backward = multiply grad by -lambda
  CameraClassifier: FC(1024->512)->BN->ReLU -> FC(512->256)->BN->ReLU -> FC(256->4)

WHY no Dropout, WHY default init:
  Classifier must first LEARN cameras (cam_acc ~90%), then GRL fights it down.
  Default kaiming_uniform init ensures fast learning from epoch 1.
  Dropout prevents the classifier from learning cameras at all.
"""

import torch
import torch.nn as nn


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
    Output: (B, num_cameras) camera logits.

    FC(1024->512)->BN->ReLU -> FC(512->256)->BN->ReLU -> FC(256->num_cameras)
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
        # Default PyTorch kaiming_uniform init — no override needed

    def forward(self, x):
        return self.classifier(x)