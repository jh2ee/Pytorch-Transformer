import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionLayer(nn.Module):


    def __init__(self, d_model, h, qkv_fc, out_fc): # qkv_fc : query, key, value의 FC
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        # 각 FC가 별개 값 가지기 위해 copy
        self.q_fc = copy.deepcopy(qkv_fc)
        self.k_fc = copy.deepcopy(qkv_fc)
        self.v_fc = copy.deepcopy(qkv_fc)
        self.out_fc = out_fc

    
    def forward(self, query, key, value, mask=None):
        # query, key, value : (n_batch * seq_len * d_embed)
        # mask : (n_batch * seq_len * seq_len)
        # return value : (n_batch * h * seq_len * d_k)
        n_batch = query.size(0)


        def calc_attention(self, query, key, value, mask):
            # query, key, value : (n_batch * h * seq_len * d_k)
            # mask : (n_batch * 1 * seq_len * seq_len)
            d_k = key.shape[-1]

            # (n_batch * h * seq_len * seq_len)
            attention_score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 

            if mask is not None:
                attention_score = attention_score.masked_fill(mask==0, -1e9)
            attention_prob = F.softmax(attention_score, dim=-1) # (n_batch * seq_len * h * seq_len)
            out = torch.matmul(attention_prob, value) # (n_batch * h * seq_len * d_k)
            return out


        def transform(x, fc): # (n_batch * seq_len * d_embed)
            out = fc(x)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch * seq_len * h * d_k)
            out = out.transpose(1, 2) # (n_batch * h * seq_len * d_k)
            return out

        
        query = transform(query, self.q_fc)
        key = transform(key, self.k_fc)
        value = transform(value, self.v_fc)

        out = self.calc_attention(query, key, value, mask) # (n_batch * h * seq_len * d_k)
        out = out.transpose(1, 2) # (n_batch * seq_len * h * d_k)

        # h와 d_k 병합 : (n_batch * seq_len * d_model)
        out = out.contiguous().view(n_batch, -1, self.d_model) 
        out = self.out_fc(out) # (n_batch * seq_len * d_embed)
        return out