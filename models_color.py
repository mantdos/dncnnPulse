import torch
import torch.nn as nn
import dncnn.basicblock as B
class DnCNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=20, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DnCNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        self.model = B.sequential(m_head, *m_body, m_tail)

    def forward(self, x):
        with torch.no_grad():
            n = self.model(x)
        return n