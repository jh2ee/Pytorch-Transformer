import torch
import torch.nn as nn

class Encoder(nn.Module):


    def __init__(self, encoder_block, n_layer): # n_layer : layer 수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            # n_layer 만큼 layer 추가(복사)
            self.layers.append(copy.deepcopy(encoder_block))

        
    def forward(self, x, mask):
        out = x
        for layer in self.layers:
            # 이전 layer의 출력을 다음 layer에 입력
            out = layer(out, mask)
        return out