import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class Mamba2MacOS(nn.Module):
    """
    A macOS-compatible implementation of the Mamba2 architecture.
    This version implements the core Mamba2 features without Triton dependencies.
    """
    
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=128,
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        activation="silu",
        bias=False,
        conv_bias=True,
        chunk_size=256,
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        # Model dimensions
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.headdim = headdim
        self.ngroups = ngroups
        self.nheads = self.d_inner // headdim
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.activation = activation
        
        # Input projection (combines x and z streams)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.rand(self.d_inner, d_state, **factory_kwargs) * 
                                          (A_init_range[1] - A_init_range[0]) + A_init_range[0]))
        self.D = nn.Parameter(torch.ones(self.d_inner, **factory_kwargs))
        
        # Delta time projection
        self.dt_proj = nn.Linear(
            self.d_inner,
            self.nheads + 2 * self.d_state,
            bias=True,
            **factory_kwargs,
        )
                
        # Initialize dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)
        
        # Activation function
        self.act = nn.SiLU() if activation == "silu" else nn.SiLU()
        
        # Layer normalization
        self.norm = nn.LayerNorm(self.d_inner, eps=1e-5, **factory_kwargs)
    
    def forward(self, hidden_states, inference_params=None, seq_idx=None, cu_seqlens=None):
        """
        Forward pass of the Mamba2 block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, d_model)
            inference_params: Optional inference parameters for caching states
            seq_idx: Optional sequence indices for variable length sequences
            cu_seqlens: Optional cumulative sequence lengths for variable length sequences
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project input to get x and z streams
        xz = self.in_proj(hidden_states)  # (batch_size, seq_len, d_inner*2)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch_size, seq_len, d_inner)
        
        # Apply convolution
        x = x.transpose(1, 2)  # (batch_size, d_inner, seq_len)
        x = self.conv1d(x)[..., :seq_len]  # (batch_size, d_inner, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_inner)
        x = self.act(x)
        
        # Project to get delta, B, and C
        x_proj = self.dt_proj(x)  # (batch_size, seq_len, d_inner)
        dt, B, C = torch.split(x_proj, [self.nheads, self.d_state, self.d_state], dim=-1)
        
        # Process delta
        dt = F.softplus(dt + self.dt_bias)  # (batch_size, seq_len, nheads)
        
        # Get A matrix
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Reshape for selective scan
        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l n -> b l 1 n")
        C = rearrange(C, "b l n -> b l 1 n")
        
        # Apply selective scan
        y = selective_scan_fn(
            x, dt, A, B, C, self.D,
            z=z,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=inference_params is not None
        )
        
        if inference_params is not None:
            y, last_state = y
            if self.layer_idx is not None:
                inference_params.key_value_memory_dict[self.layer_idx] = last_state
        
        # Reshape back
        y = rearrange(y, "b l h p -> b l (h p)")
        
        # Apply normalization and gate
        y = self.norm(y)
        y = y * torch.sigmoid(z)
        
        # Project to output dimension
        output = self.out_proj(y)
        
        return output
    
    def step(self, hidden_states, conv_state, ssm_state):
        """
        Step function for autoregressive generation.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, 1, d_model)
            conv_state: Convolution state
            ssm_state: SSM state
            
        Returns:
            Tuple of (output, new_conv_state, new_ssm_state)
        """
        batch_size = hidden_states.shape[0]
        
        # Project input
        xz = self.in_proj(hidden_states.squeeze(1))  # (batch_size, d_inner*2)
        x, z = xz.chunk(2, dim=-1)  # Each: (batch_size, d_inner)
        
        # Update conv state
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x
        x = torch.sum(conv_state * self.conv1d.weight.squeeze(1), dim=-1)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = self.act(x)
        
        # Project to get delta, B, and C
        x_proj = self.dt_proj(x)  # (batch_size, d_inner)
        dt, B, C = torch.split(x_proj, [self.nheads, self.d_state, self.d_state], dim=-1)
        
        # Process delta
        dt = F.softplus(dt + self.dt_bias)  # (batch_size, nheads)
        
        # Get A matrix
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Reshape for selective scan
        x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        B = rearrange(B, "b n -> b 1 n")
        C = rearrange(C, "b n -> b 1 n")
        
        # Update SSM state
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (batch_size, nheads, d_state)
        ssm_state = ssm_state * dA + torch.einsum("bh,bn->bhn", B, x)
        y = torch.einsum("bhn,bn->bh", ssm_state, C)
        y = y + self.D.view(self.nheads, self.headdim) * x
        
        # Reshape back
        y = rearrange(y, "b h p -> b (h p)")
        
        # Apply normalization and gate
        y = self.norm(y)
        y = y * torch.sigmoid(z)
        
        # Project to output dimension
        output = self.out_proj(y)
        
        return output.unsqueeze(1), conv_state, ssm_state
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate cache for inference.
        
        Args:
            batch_size: Batch size
            max_seqlen: Maximum sequence length
            dtype: Optional dtype for the cache
            
        Returns:
            Tuple of (conv_state, ssm_state)
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_inner, self.d_conv,
            device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.d_state,
            device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state 