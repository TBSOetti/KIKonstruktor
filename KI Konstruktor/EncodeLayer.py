import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from SAttention import MultiHeadAttention
from FForward import Feedforward


#ReadMe TransformerEncoderLayer
class TransformerEncoderLayer(nn.Module): 
    def __init__(self, d_model,num_heads,d_ff,dropout=0.1): 
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = Feedforward(d_model,d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    #ReadMe src 
    def forward(self,src,mask=None): 
        src2 = self.norm1(src)
        src = src + self.dropout(self.self_attn(src2,src2,src2,mask))
        src2 = self.norm2(src)
        src = src + self.dropout(self.feed_forward(src2))
        return src
