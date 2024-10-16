import torch.nn as nn
import torch

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class MyAttention(nn.Module):
    '''
    Modified version of Attention
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        old_attn = (q @ k.transpose(-2, -1)) * self.scale
        old_attn = old_attn.softmax(dim=-1)
        attn = self.attn_drop(old_attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.mean(dim=-1) #B N C-> B N
        return x, old_attn

class MySimpleAttention(nn.Module):
    '''
    Directly use mlp instead of attention
    B N D 
    '''
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        
    def forward(self, x):
        old_x = x
        # B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B,N,3,C).permute(2,0,1,3) # 3 B N C
        # q, k, v = qkv[0], qkv[1], qkv[2] #B N C
        q,k,v = x, x, x

        atten = (q @ k.transpose(-2,-1)).softmax(dim=-1) #B N N
        
        x = atten @ v #B N C

        x = old_x + x
        x = x.mean(1) #使用均值，如果使用x[0]的话，会极度不收敛。差别就是这么大 ; 如果不使用qkv的变换的话，那么效果也不拟合
        #return x, (old_x@old_x.transpose(-2,-1))[:,0]
        return x, atten

class MyChannelAttention(nn.Module):
    '''
    Use channel attention
    B N D 
    '''
    def __init__(self,dim):
        super().__init__()
        self.v = nn.Linear(dim, dim)
    
    def forward(self, x):
        old_x = x
        v = self.v(x)

        atten = (x[:,0][:, None, :] @ x[:,1:].transpose(-2,-1)).softmax(dim=-1) #B 1 N-1
        
        x = torch.mul(v[:,1:] ,atten.transpose(-2,-1))

        #x = old_x[:,1:] + x
        x = x.mean(-1)
        #return x, (old_x@old_x.transpose(-2,-1))[:,0]
        return x, atten