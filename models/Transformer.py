import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    

    def __init__(self, src_embed, tgt_embed, encoder, decoder, generator):
        super(Transformer, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x, mask):
        out = self.encoder(self.src_embed(x), mask)
        return out


    def decode(self, z, c, tgt_mask, src_tgt_mask):
        out = self.decoder(self.tgt_embed(z), c, tgt_mask, src_tgt_mask)
        return out

    
    def forward(self, x, z):
        src_mask = self.make_src_mask(x)
        tgt_mask = self.make_tgt_mask(x)
        src_tgt_mask = self.make_src_tgt_mask(x, z)
        c = self.encode(x, src_mask)
        y = self.decode(z, c, tgt_mask, src_mask)
        out = self.generator(y)
        out = F.log_softmax(out, dim=1)
        return out, y

    
    def make_pad_mask(self, query, key, pad_idx=1):
        # query : (n_batch * query_seq_len)
        # key : (n_batch, key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        # (n_batch * 1 * 1 * key_seq_len)
        # ne(pad_idx) : pad_idx와 같지 않은 값을 true, 같은 값 false로 하는 mask 생성
        # unsqueeze(1).unsqueeze(2) : mask 값을 1, 2번째 차원 추가
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)
        # (n_batch * 1 * query_seq_len * key_seq_len)
        # 3번째 차원의 값 query_seq_len으로 변경
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)

        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch * 1 * query_seq_len * 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch * 1 * query_seq_len * key_seq_len)

        mask = key_mask & query_mask # AND 연산으로 mask 생성
        mask.requires_grad = False # backpropagation 중 변경되지 않도록 값 고정
        return mask


    def make_src_mask(self, src): # self-attention
        pad_mask = self.make_pad_mask(src, src)
        return pad_mask


    def make_subsequent_mask(query, key): # teacher forcing : i번째 token 생성 시 1~i-1번째 토큰 안보이도록 처리
        # query : (n_batch * query_seq_len)
        # key : (n_batch * key_seq_len)
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8')
        mask = torch.tensor(tril, dtype=torch.book, requires_grad=False, device=query.device)
        return mask


    def make_tgt_mask(self, x):
        pad_mask = self.make_pad_mask(x, x)
        seq_mask = self.make_subsequent_mask(x, x)
        mask = pad_mask & seq_mask
        return mask


    def make_src_tgt_mask(self, x, z):
        pad_mask = self.make_pad_mask(z, x)
        return pad_mask