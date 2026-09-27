"""Gradient reversal layer and camera classifier for domain-adversarial training.

The camera classifier predicts which camera an image came from. The gradient
reversal layer passes features through unchanged in the forward pass and
multiplies the gradient by -lambda in the backward pass, so the backbone is
pushed towards embeddings that do not encode the camera.
"""

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity in the forward pass, gradient scaled by -lambda in the backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        # .item() keeps the dtype compatible with mixed-precision gradients
        return -grad_output * lambda_.item(), None


def grad_reverse(x, lambda_=1.0):
    """Apply gradient reversal with weight lambda_."""
    lambda_tensor = torch.tensor(lambda_, dtype=torch.float32).to(x.device)
    return GradientReversalFunction.apply(x, lambda_tensor)


class CameraClassifier(nn.Module):
    """3-layer MLP camera classifier.

    Input:  (B, in_dim) CLS token before the BN neck.
    Output: (B, num_cameras) logits.
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

    def forward(self, x):
        return self.classifier(x)
