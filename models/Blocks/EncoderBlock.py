import torch
import torch.nn as nn

class EncoderBlock(nn.Module):


    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for i in range(2)]

    
    def forward(self, x, mask):
        out = x
        out = self.residuals[0](out, lambda out : self.self_attention(query=out, key=out, value=out, mask=mask))
        out = self.residuals[1](out, self.position_ff)
        return out