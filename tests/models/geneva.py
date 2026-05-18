import math
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import functional as F
from torch.nn import GELU, ReLU, Tanh, Sigmoid
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
try: 
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
except: pass

#from data_utils.utils import MultipleTensors
class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item]
    
ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU()}

'''
    A simple MLP class, includes at least 2 layers and n hidden layers
'''
class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', gating=False):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])
        self.gating = gating

        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])

    def forward(self, x, **kwargs):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            x = self.act(self.linears[i](x)) + x
            # x = self.act(self.bns[i](self.linears[i](x))) + x

        x = self.linear_post(x)
        return x
    
class rollout_RNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_layers=1, act='gelu', gating=False):
        super(rollout_RNN, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.rnn = nn.RNN(n_hidden, n_hidden, num_layers=n_layers, nonlinearity='tanh', bias=False, batch_first=True)
        self.raise_dim = nn.Linear(n_input, n_hidden)

        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])

    def forward(self, input_f, masking=None):
        hn = None
        x = self.raise_dim(input_f)
        for i in range(input_f.shape[1]):
            #x = self.raise_dim(x[:,i,:,:])#*(1 if masking is None else masking[:,i].reshape(-1, 1, 1))
            output, hn = self.rnn(x[:,i,:,:], hn)
            hn = self.act(hn)#*(1 if masking is None else masking[:,i].reshape(-1, 1, 1)))
        x = self.act(output)
        return x
    
class encoder_propagator(nn.Module):
    def __init__(self, n_input, n_hidden_prop, n_layers=1):
        super(encoder_propagator, self).__init__()

        # Timeseries encoder: (to replace RNN)
        #self.input_dim_raiser = MLP(self.output_size, n_hidden, n_hidden_prop, n_layers=mlp_layers,act=act)
        self.n_layers = n_layers
        self.n_hidden_prop = n_hidden_prop 
        self.expand_feat2 = nn.Linear(n_input, n_hidden_prop)
        self.input_encoder = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(n_hidden_prop),
               nn.Sequential(
                    nn.Linear(n_hidden_prop*2, n_hidden_prop, bias=False),
                    nn.GELU(),
                    nn.Linear(n_hidden_prop, n_hidden_prop, bias=False),
                    nn.GELU(),
                    nn.Linear(n_hidden_prop, n_hidden_prop, bias=False))])
            for _ in range(n_layers)])
        
    def forward(self, input_solution, masking=None):
        
        # Init H0
        B = input_solution.shape[0]
        T = input_solution.shape[1]
        N = input_solution.shape[2]
        h = torch.zeros(B, N,  self.n_hidden_prop).to(input_solution.device)

        
        # For masking and each timestep, if time_step is padded, we zero that batch
        # output is [batch, nodes, embedding dim] * masking where masking is over batch dim

        # Propagate Hidden state along time dim
        for step in range(input_solution.shape[1]):
            for layer in self.input_encoder:
                norm_fn, ffn = layer
                x = self.expand_feat2(input_solution[:,step,...])
                h = h + (ffn(torch.concat((norm_fn(h), x), dim=-1)))*(1 if masking is None else masking[:,step].reshape(-1, 1, 1))
        return h


class MoEGPTConfig():
    """ base GPT config, params common to all GPT versions """
    def __init__(self,attn_type='linear', embd_pdrop=0.0, resid_pdrop=0.0,attn_pdrop=0.0, n_embd=128, n_head=1, n_layer=3, block_size=128, n_inner=4,act='gelu',n_experts=2,space_dim=1,branch_sizes=None,n_inputs=1, rnn_input_i=None):
        self.attn_type = attn_type
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.n_embd = n_embd  # 64
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.n_inner = n_inner * self.n_embd
        self.act = act
        self.n_experts = n_experts
        self.space_dim = space_dim
        self.branch_sizes = branch_sizes
        self.n_inputs = n_inputs

        # For Flash Attention
        self.causal = False
        self.window_size = (-1, -1)

        # For RNN Rollout of Input
        self.rnn_input_i = rnn_input_i

class FlashAttention(nn.Module):
    """
    Using Flash Attention V2
    Features Cross and Self-Attention
    """

    def __init__(self, config, cross=False, attention_fn='flash'):
        super(FlashAttention, self).__init__()
        assert config.n_embd % config.n_head == 0

        self.cross = cross
        # key, query, value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        if self.cross:
            self.keys = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])
            self.values = nn.ModuleList([nn.Linear(config.n_embd, config.n_embd) for _ in range(config.n_inputs)])

        else:
            self.keys = nn.Linear(config.n_embd, config.n_embd)
            self.value = nn.Linear(config.n_embd, config.n_embd)
        
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_inputs = config.n_inputs
        self.causal = config.causal
        self.window_size = config.window_size
        self.dropout_p = config.attn_pdrop

        # hardcoded manual switch for inference
        self.attention_fn = attention_fn

    '''
        Linear Attention and Linear Cross Attention
    '''
    def forward(self, x, y=None):
        # masking has size ([batch_size, 1, 1, seq_len])
        # This is not causal, only for batch padding
        
        #y = x if y is None else y
        B, T1, C = x.size()
        
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(x).view(B, T1, self.n_head, C // self.n_head)
        out = q

        if self.cross and y is not None:
            for i in range(self.n_inputs):
                T2 = y[i].shape[-2]
                k = self.keys[i](y[i]).view(B, T2, self.n_head, C // self.n_head).bfloat16()
                v = self.values[i](y[i]).view(B, T2, self.n_head, C // self.n_head).bfloat16()
                if self.attention_fn == 'flash':
                    out = out + flash_attn_func(q.bfloat16(), k, v, self.dropout_p, causal=self.causal, window_size=self.window_size, deterministic=False).float()
                elif self.attention_fn == 'base':
                    out = out + torch.nn.functional.scaled_dot_product_attention(q.bfloat16(), k, v, dropout_p=self.dropout_p, is_causal=self.causal, scale=None, enable_gqa=False).float()
        else:
            k = self.keys(x).view(B, T1, self.n_head, C // self.n_head).bfloat16()
            v = self.value(x).view(B, T1, self.n_head, C // self.n_head).bfloat16()
            if self.attention_fn == 'flash':
                out = flash_attn_func(q.bfloat16(), k, v, self.dropout_p, causal=self.causal, window_size=self.window_size, deterministic=False).float()
            elif self.attention_fn == 'base':
                out = out + torch.nn.functional.scaled_dot_product_attention(q.bfloat16(), k, v, dropout_p=self.dropout_p, is_causal=self.causal, scale=None, enable_gqa=False).float()
        
        # output projection
        out = rearrange(out, 'b n h d -> b n (h d)')
        out = self.proj(out)
        return out
    
'''
Self and Cross Attention block for CGPT, contains  a cross attention block and a self attention block
'''
class MIOECrossAttentionBlock(nn.Module):
    def __init__(self, config):
        super(MIOECrossAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2_branch = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(config.n_inputs)])
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.ln4 = nn.LayerNorm(config.n_embd)
        self.ln5 = nn.LayerNorm(config.n_embd)
        if config.attn_type in ['flash','base']:
            self.selfattn = FlashAttention(config, attention_fn=config.attn_type)
            self.crossattn = FlashAttention(config, cross=True, attention_fn=config.attn_type)
        else:
            raise NotImplementedError('Geneva currently only supports Flash Attention')

        if config.act == 'gelu':
            self.act = GELU
        elif config.act == "tanh":
            self.act = Tanh
        elif config.act == 'relu':
            self.act = ReLU
        elif config.act == 'sigmoid':
            self.act = Sigmoid

        self.resid_drop1 = nn.Dropout(config.resid_pdrop)
        self.resid_drop2 = nn.Dropout(config.resid_pdrop)

        self.n_experts = config.n_experts
        self.n_inputs = config.n_inputs

        self.moe_mlp1 = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        ) for _ in range(self.n_experts)])

        self.moe_mlp2 = nn.ModuleList([nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_embd),
        ) for _ in range(self.n_experts)])

        self.gatenet = nn.Sequential(
            nn.Linear(config.space_dim, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, config.n_inner),
            self.act(),
            nn.Linear(config.n_inner, self.n_experts)
        )


    def ln_branchs(self, y):
        return MultipleTensors([self.ln2_branch[i](y[i]) for i in range(self.n_inputs)])

    '''
        x: [B, T1, C], y:[B, T2, C], pos:[B, T1, n]
    '''
    def forward(self, x, y, pos):
        gate_score = F.softmax(self.gatenet(pos),dim=-1).unsqueeze(2)    # B, T1, 1, m
        
        x = x + self.resid_drop1(self.crossattn(self.ln1(x), self.ln_branchs(y)))
        
        x_moe1 = torch.stack([self.moe_mlp1[i](x) for i in range(self.n_experts)],dim=-1) # B, T1, C, m
        x_moe1 = (gate_score*x_moe1).sum(dim=-1,keepdim=False)
       
        x = x + self.ln3(x_moe1)
        x = x + self.resid_drop2(self.selfattn(self.ln4(x)))
        
        x_moe2 = torch.stack([self.moe_mlp2[i](x) for i in range(self.n_experts)],dim=-1) # B, T1, C, m
        x_moe2 = (gate_score*x_moe2).sum(dim=-1,keepdim=False)
        
        x = x + self.ln5(x_moe2)
        return x
    
'''
Cross Attention GPT neural operator
Trunck Net: geom + RNN
'''

class GenevaNOT(nn.Module):
    def __init__(self,
                 trunk_size=2,
                 branch_sizes=[1],
                 space_dim=2,
                 time_dim=None,
                 output_size=3,
                 n_layers=3,
                 n_hidden=64,
                 n_head=1,
                 n_experts = 2,
                 n_inner = 4,
                 mlp_layers=2,
                 attn_type='linear',
                 act = 'gelu',
                 ffn_dropout=0.0,
                 attn_dropout=0.0,
                 horiz_fourier_dim = 0,
                 gating = True,
                 rnn_input_i=None,
                 **kwargs
                 ):
        super(GenevaNOT, self).__init__()

        # For RNN
        self.rnn_input_i = rnn_input_i
    
        self.gating = gating
        self.horiz_fourier_dim = horiz_fourier_dim
        self.trunk_size = trunk_size * (4*horiz_fourier_dim + 3) if horiz_fourier_dim>0 else trunk_size
        self.branch_sizes = [bsize * (4*horiz_fourier_dim + 3) for bsize in branch_sizes] if horiz_fourier_dim > 0 else branch_sizes
        self.n_inputs = len(self.branch_sizes)
        self.output_size = output_size
        self.space_dim = space_dim

        # For Decoder and Propagator
        self.decoding_depth = 1
        self.rolling_checkpoint = False # trades memory for compute time (does not work with grad)
        n_hidden_prop = int(2*n_hidden) # expand features

        # Get Layers
        self.gpt_config = MoEGPTConfig(attn_type=attn_type,embd_pdrop=ffn_dropout, resid_pdrop=ffn_dropout, attn_pdrop=attn_dropout,n_embd=n_hidden, n_head=n_head, n_layer=n_layers,
                                       block_size=128,act=act, n_experts=n_experts,space_dim=space_dim, branch_sizes=branch_sizes,n_inputs=len(branch_sizes),n_inner=n_inner,
                                       rnn_input_i=rnn_input_i)
  
        self.trunk_mlp = MLP(self.trunk_size, n_hidden, n_hidden, n_layers=mlp_layers,act=act)
        self.branch_mlps = nn.ModuleList([MLP(bsize, n_hidden, n_hidden, n_layers=mlp_layers,act=act) for bsize in self.branch_sizes])

        # override for index of input_f that is a timeseries
        if rnn_input_i is not None:
            self.branch_mlps[rnn_input_i] = rollout_RNN(self.branch_sizes[rnn_input_i], n_hidden)
            #self.branch_mlps[rnn_input_i] = encoder_propagator(self.branch_sizes[rnn_input_i], n_hidden, n_layers=1)

        self.blocks = nn.Sequential(*[MIOECrossAttentionBlock(self.gpt_config) for _ in range(self.gpt_config.n_layer)])

        self.out_mlp = MLP(n_hidden, n_hidden, output_size, n_layers=mlp_layers)
        
        
        
        # Decoder Layers
        self.expand_feat = nn.Linear(n_hidden, n_hidden_prop)
        
        # Postion disclaimer
        position_prop_adjust = 1
        assert position_prop_adjust == 1, 'Currently propagation of xy-coordinates (2) or +time (3) not implemented, only time (1)'
        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(n_hidden_prop),
               nn.Sequential(
                    nn.Linear(n_hidden_prop + position_prop_adjust, n_hidden_prop, bias=False),
                    nn.GELU(),
                    nn.Linear(n_hidden_prop, n_hidden_prop, bias=False),
                    nn.GELU(),
                    nn.Linear(n_hidden_prop, n_hidden_prop, bias=False))])
            for _ in range(self.decoding_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(n_hidden_prop),
            nn.Linear(n_hidden_prop, n_hidden, bias=False),
            nn.GELU(),
            nn.Linear(n_hidden, n_hidden, bias=False),
            nn.GELU(),
            nn.Linear(n_hidden, self.output_size, bias=True))
        
        # self.apply(self._init_weights)

        self.__name__ = 'MIOEGPT_Geneva'


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.0002)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def propagate(self, h, propagate_pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            h = h + ffn(torch.concat((norm_fn(h), propagate_pos), dim=-1))
        return h
    
    # def encode(self, h, input_solution):
    #     '''
    #     h = coordinates
    #     input_solution = initil time-steps at those coordinates
    #     '''
    #     h = self.expand_feat2(h)
    #     for step in range(input_solution.shape[1]):
    #         for layer in self.propagator:
    #             norm_fn, ffn = layer
    #             h = h + ffn(torch.concat((norm_fn(h), input_solution[:,step,...]), dim=-1))
    #     return h

    def decode(self, h):
        h = self.to_out(h)
        return h

    def rollout(self, h, time_steps):

        history = []
        
        h = self.expand_feat(h)
        forward_steps = time_steps.shape[1]
        propagate_pos = time_steps.repeat(1,1,h.shape[1]).permute(0,2,1)
        
        for step in range(forward_steps):
            if self.rolling_checkpoint and self.training:
                h = checkpoint(self.propagate, h, propagate_pos[...,[step]])
                h_out = checkpoint(self.decode, h)
            else:
                h = self.propagate(h, propagate_pos[...,[step]])
                h_out = self.decode(h).unsqueeze(1)
            history.append(h_out)

            x = torch.cat(history, axis=1) # concate along time dim
        return x

    def forward(self, x, inputs=None, x_t=None , initial_condition=None, masking=None):
        
        if self.gating:
            pos = x[:,:,0:self.space_dim]
        else:
            pos = None
        
        # if self.horiz_fourier_dim > 0:
        #     x = horizontal_fourier_embedding(x, self.horiz_fourier_dim)
        #     z = horizontal_fourier_embedding(z, self.horiz_fourier_dim)

        x = self.trunk_mlp(x)
        if self.n_inputs:
            z = MultipleTensors([self.branch_mlps[i](inputs[i], masking=(masking[i] if masking is not None else None)) for i in range(self.n_inputs)])
        else:
            z = MultipleTensors([x])

        for block in self.blocks:
            x = block(x, z, pos)
        
        if x_t is not None:
            x = self.rollout(h=x, time_steps=x_t)
        else:
            x = self.out_mlp(x)

        return x
    
