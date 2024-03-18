import torch.nn as nn


class Decoder(nn.Module):
    

    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.Module([copy.deepcopy(decoder_block) for i in range(self.n_layer)])


    def forward(self, x, y, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, y, tgt_mask, src_tgt_mask)
            
        return out