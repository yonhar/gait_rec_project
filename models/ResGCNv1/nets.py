import torch
from torch import nn
import torch.nn.functional as F

from .modules import ResGCN_Module


class ResGCN_Input_Branch(nn.Module):
    def __init__(self, structure, block, num_channel, A, **kwargs):
        super(ResGCN_Input_Branch, self).__init__()

        self.register_buffer('A', A)

        module_list = [ResGCN_Module(num_channel, 64, 'Basic', A, initial=True, **kwargs)]
        module_list += [ResGCN_Module(64, 64, 'Basic', A, initial=True, **kwargs) for _ in range(structure[0] - 1)]
        module_list += [ResGCN_Module(64, 64, block, A, **kwargs) for _ in range(structure[1] - 1)]
        module_list += [ResGCN_Module(64, 32, block, A, **kwargs)]

        self.bn = nn.BatchNorm2d(num_channel)
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        x = self.bn(x)
        for layer in self.layers:
            x = layer(x, self.A)

        return x

class ResGCN(nn.Module):
    def __init__(self, module, structure, block, num_input, num_channel, num_class, A, **kwargs):
        super(ResGCN, self).__init__()

        self.register_buffer('A', A)

        # input branches
        self.input_branches = nn.ModuleList([
            ResGCN_Input_Branch(structure, block, num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        # main stream
        module_list = [module(32*num_input, 128, block, A, stride=2, **kwargs)]
        module_list += [module(128, 128, block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, block, A, stride=2, **kwargs)]
        module_list += [module(256, 256, block, A, **kwargs) for _ in range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)

        # output
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def forward(self, x):

        # N, I, C, T, V = x.size()

        # input branches
        x_cat = []
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:,i,:,:,:]))
        x = torch.cat(x_cat, dim=1)

        # main stream
        for layer in self.main_stream:
            x = layer(x, self.A)
        
        # output
        x = self.global_pooling(x)
        x = self.fcn(x.squeeze())

        # L2 normalization
        x = F.normalize(x, dim=1, p=2)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ResGCN_transformer(nn.Module):
    def __init__(self, module, structure, block, num_input, num_channel, num_class, A, **kwargs):
        super(ResGCN_transformer, self).__init__()

        self.register_buffer('A', A)
        self.input_branches = nn.ModuleList([
            ResGCN_Input_Branch(structure, block, num_channel, A, **kwargs)
            for _ in range(num_input)
        ])

        module_list = [module(32*num_input, 128, block, A, stride=1, **kwargs)]
        module_list += [module(128, 128, block, A, **kwargs) for _ in range(structure[2] - 1)]
        module_list += [module(128, 256, block, A, stride=1, **kwargs)]
        module_list += [module(256, 256, block, A, **kwargs) for _ in range(structure[3] - 1)]
        self.main_stream = nn.ModuleList(module_list)

        self.feature_dim = 256
        
        self.num_heads = 8
        self.head_dim = self.feature_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.k_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.v_proj = nn.Linear(self.feature_dim, self.feature_dim)
        self.out_proj = nn.Linear(self.feature_dim, self.feature_dim)
        
        self.ffn1 = nn.Linear(self.feature_dim, 1024)
        self.ffn2 = nn.Linear(1024, self.feature_dim)
        
        self.norm1 = nn.LayerNorm(self.feature_dim)
        self.norm2 = nn.LayerNorm(self.feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        
        self.fcn = nn.Linear(256, num_class)

        # init parameters
        init_param(self.modules())
        zero_init_lastBN(self.modules())

    def _attention_block(self, x):
        residual = x
        
        x = self.norm1(x)
        
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights  = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_weights , dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.feature_dim)
        
        output = self.out_proj(context)
        output = self.dropout(output)
        
        output = output + residual
        
        residual = output
        output = self.norm2(output)
        output = self.ffn1(output)
        output = F.relu(output)
        output = self.dropout(output)
        output = self.ffn2(output)
        output = self.dropout(output)
        output = output + residual
        
        return output
    
    def transformer_encoder(self, x):
        x = self.positional_encoding(x)
        x = self._attention_block(x)
        x = self._attention_block(x)
        return x

    def forward(self, x):
        x_cat = []
        for i, branch in enumerate(self.input_branches):
            x_cat.append(branch(x[:, i, :, :, :]))
        x = torch.cat(x_cat, dim=1)

        for layer in self.main_stream:
            x = layer(x, self.A)
        
        x = torch.mean(x, dim=3)
        
        x = x.permute(0, 2, 1)
        
        x = self.transformer_encoder(x)
        
        x = torch.mean(x, dim=1)
        
        x = self.fcn(x)
        
        x = F.normalize(x, p=2, dim=1)
        
        return x

def init_param(modules):
    for m in modules:
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            #m.bias = None
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def zero_init_lastBN(modules):
    for m in modules:
        if isinstance(m, ResGCN_Module):
            if hasattr(m.scn, 'bn_up'):
                nn.init.constant_(m.scn.bn_up.weight, 0)
            if hasattr(m.tcn, 'bn_up'):
                nn.init.constant_(m.tcn.bn_up.weight, 0)
