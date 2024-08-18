import torch
import torch.nn as nn 
import torch.nn.functional as F
#ReadMe MultiHead 
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads):
        super(MultiHeadAttention, self).__init__()
        #ReadMe assert d_model
        assert d_model % num_heads == 0 
        self.d_k = d_model // num_heads 
        self.num_heads = num_heads 
        self.linear_q = nn.Linear(d_model,d_model)
        self.linear_k = nn.Linear(d_model,d_model)
        self.linear_v = nn.Linear(d_model,d_model)
        self.linear_out = nn.Linear(d_model,d_model)

    #ReadMe MHA
    def forward(self,query,key,value,mask = None):
        batch_size = query.size(0)
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        key = self.linear_k(key).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        value = self.linear_v(value).view(batch_size,-1,self.num_heads,self.d_k).transpose(1,2)
        scores = torch.matmul(query,key.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.d_k,dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores,dim=-1)
        x = torch.matmul(attention, value).transpose(1,2).contiguous().view(batch_size,-1, self.num_heads * self.d_k)
        return self.linear_out(x)