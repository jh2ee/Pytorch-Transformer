import torch.nn as nn


class DecoderBlock(nn.Module):


    def __init__(self, self_attention, cross_attention, position_ff):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for i in range(3)]


    def forward(self, z, c, tgt_mask, src_tgt_mask):
        out = z
        out = self.residuals[0](out, lambda out : self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals[1](out, lambda out : self.cross_attention(query=out, key=c, value=c, mask=src_tgt_mask))
        out = self.residuals[2](out, self.position_ff)
        return out